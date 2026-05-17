import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from body_model import OP_IGNORE_JOINTS
from util.logger import Logger, log_cur_stats
from util.tensor import move_to, detach_all
from vis.output import prep_result_vis, animate_scene

from .losses import RootLoss, SMPLLoss

LINE_SEARCH = "strong_wolfe"

"""
Optimization happens in multiple stages
Stage 1: fit root orients and trans of each frame independently
Stage 2: fit SMPL poses and betas of each frame independently
"""


class StageOptimizer(object):
    def __init__(
        self,
        name,
        model,
        param_names,
        lr=1.0,
        lbfgs_max_iter=20,
        save_every=10,
        vis_every=-1,
        save_meshes=True,
        max_chunk_steps=10,
        **kwargs,
    ):
        Logger.log(f"INITIALIZING OPTIMIZER {name} for {param_names}")
        self.name = name
        self.model = model

        self.set_opt_vars(param_names)

        self.optim = torch.optim.LBFGS(
            self.opt_params, max_iter=lbfgs_max_iter, lr=lr, line_search_fn=LINE_SEARCH
        )
        # LBFGS computes losses multiple times per iteration,
        # save a dict mapping iteration to list of stats_dicts
        self.loss_dicts = {}

        self.save_every = save_every
        self.vis_every = vis_every
        self.save_meshes = save_meshes
        self.max_chunk_steps = max_chunk_steps

        self.cur_step = 0

        self.add_chunk = 0
        self.cur_loss = 0
        self.prev_loss = np.inf
        self.last_updated = 0
        self.reached_max = False
        self.reached_max_iter = -1

    def set_opt_vars(self, param_names):
        Logger.log("Set param names:")
        Logger.log(param_names)

        self.param_names = param_names
        self.model.params.set_require_grads(self.param_names)
        self.opt_params = [
            getattr(self.model.params, name) for name in self.param_names
        ]

    def forward_pass(self, obs_data):
        raise NotImplementedError

    def load_checkpoint(self, out_dir, device=None):
        if device is None:
            device = torch.device("cpu")

        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        if os.path.isfile(param_path):
            param_dict = torch.load(param_path, map_location=device)
            self.model.params.load_dict(param_dict)
            Logger.log(f"Params loaded from {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        if os.path.isfile(optim_path):
            optim_dict = torch.load(optim_path)
            self.optim.load_state_dict(optim_dict["optim"])
            self.cur_step = optim_dict["cur_step"]
            Logger.log(f"Optimizer loaded from {optim_path} at iter {self.cur_step}")

    def save_checkpoint(self, out_dir):
        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        param_dict = self.model.params.get_dict()
        if "world_scale" in param_dict:
            print("WORLD_SCALE", param_dict["world_scale"].detach().cpu())
        if "floor_plane" in param_dict:
            print("FLOOR PLANE", param_dict["floor_plane"].detach().cpu())
        if "cam_f" in param_dict:
            print("CAM_F", param_dict["cam_f"].detach().cpu())
        torch.save(param_dict, param_path)
        Logger.log(f"Model saved at {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        torch.save(
            {"optim": self.optim.state_dict(), "cur_step": self.cur_step},
            optim_path,
        )
        Logger.log(f"Optimizer saved at {optim_path}")

    def save_results(self, out_dir, seq_name):
        """
        pred dict will be a dictionary of trajectories.
        each trajectory will have params and lists of trimesh sequences
        """
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            pred_dict = self.model.get_optim_result()
        pred_dict = move_to(detach_all(pred_dict), "cpu")

        i = self.cur_step
        for name, results in pred_dict.items():
            out_path = f"{out_dir}/{seq_name}_{i:06d}_{name}_results.npz"
            Logger.log(f"saving params to {out_path}")
            np.savez(out_path, **results)

        self.plot_losses(out_dir)

    def vis_result(self, res_dir, obs_data, vis=None, num_steps=-1):
        if vis is None or self.vis_every < 0:
            return

        seq_name = obs_data["seq_name"][0]
        res_pre = f"{res_dir}/{seq_name}_opt_{self.cur_step:06d}"
        with torch.no_grad():
            pred_dict = self.model.get_optim_result(num_steps=num_steps)

        res_dict = detach_all(pred_dict["world"])
        scene_dict = move_to(
            prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                self.model.body_model,
            ),
            "cpu",
        )
        animate_scene(vis, scene_dict, res_pre, render_views=["src_cam", "above"])

    def log_losses(self, stats_dict):
        stats_dict = move_to(detach_all(stats_dict), "cpu")
        log_cur_stats(
            stats_dict,
            iter=self.cur_step,
            to_stdout=(self.cur_step % self.save_every == 0),
        )
        for loss_name, loss_val in stats_dict.items():
            loss_dict = self.loss_dicts.get(loss_name, {})
            loss_series = loss_dict.get(self.cur_step, [])
            loss_series.append(loss_val)
            loss_dict[self.cur_step] = loss_series
            self.loss_dicts[loss_name] = loss_dict

    def record_current_losses(self, writer):
        """
        record the mean of current step's loss values in tensorboard
        """
        if len(self.loss_dicts) < 1:
            return

        for loss_name, loss_dict in self.loss_dicts.items():
            loss_mean = np.mean(loss_dict[self.cur_step])
            writer.add_scalar(f"{self.name}/{loss_name}", loss_mean, self.cur_step)

    def plot_losses(self, res_dir):
        """
        plot a box plot for each BFGS iteration
        """
        if len(self.loss_dicts) < 1:
            return
        for loss_name, loss_dict in self.loss_dicts.items():
            # times (list len T)
            # loss vals (list len T of loss value lists)
            times, loss_vals = zip(*loss_dict.items())
            plt.figure()
            plt.boxplot(loss_vals, labels=times, showfliers=False)
            plt.savefig(f"{res_dir}/{loss_name}.png")

    def run(self, obs_data, num_iters, out_dir, vis=None, writer=None):
        self.cur_step = 0
        self.loss.cur_step = 0
        res_dir = os.path.join(out_dir, self.name)
        os.makedirs(res_dir, exist_ok=True)
        seq_name = obs_data["seq_name"][0]

        # try to load from checkpoint if exists
        device = obs_data["joints2d"].device
        self.load_checkpoint(out_dir, device=device)

        if self.cur_step >= num_iters:
            Logger.log(f"Current optimizer {self}")
            Logger.log(f"Checkpoint at {self.cur_step} >= {num_iters}, skipping")
            return

        Logger.log(f"OPTIMIZING {self.name} FOR {num_iters} ITERATIONS")

        self.save_results(res_dir, seq_name)

        for i in range(self.cur_step, num_iters):
            Logger.log("ITER: %d" % (i))

            if (i + 1) % self.save_every == 0:  # save before
                self.save_checkpoint(out_dir)
                self.save_results(res_dir, seq_name)
            else:
                self.save_checkpoint(out_dir)

            if (i + 1) % self.vis_every == 0:  # render
                self.vis_result(res_dir, obs_data, vis)

            self.cur_step = i
            self.loss.cur_step = i

            self.optim_step(obs_data, i, writer)

            # early termination in case of nans
            if np.isnan(self.cur_loss):
                self.load_checkpoint(out_dir, device=device)
                raise ValueError

            # termination for the last chunk
            if self.reached_max and self.reached_max_iter < 0:
                self.reached_max_iter = i - 1
            if self.reached_max and i - self.reached_max_iter >= self.max_chunk_steps:
                break

            # termination for middle chunks
            loss_change = self.prev_loss - self.cur_loss
            if self.last_updated == i - 1 and loss_change == 0:
                break
            if (
                (self.cur_loss < 0 and loss_change < 100)
                or (i - self.last_updated >= self.max_chunk_steps)
                or (loss_change < 20 and i - self.last_updated > 5)
            ):
                self.add_chunk = self.add_chunk + 1
                self.last_updated = i
            self.prev_loss = self.cur_loss

        # final save and vis step
        self.cur_step = num_iters
        self.save_checkpoint(out_dir)
        self.save_results(res_dir, seq_name)
        self.vis_result(res_dir, obs_data, vis)

    def optim_step(self, obs_data, i, writer=None):
        def closure():
            self.optim.zero_grad()
            loss, stats_dict, preds = self.forward_pass(obs_data)
            stats_dict["total"] = loss
            self.log_losses(move_to(detach_all(stats_dict), "cpu"))
            self.cur_loss = stats_dict["total"].detach().cpu().item()
            loss.backward()
            return loss

        self.optim.step(closure)
        if writer is not None:
            self.record_current_losses(writer)


class RootOptimizer(StageOptimizer):
    name = "root_fit"
    stage = 0

    def __init__(
        self,
        model,
        all_loss_weights,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        param_names = ["trans", "root_orient"]
        super().__init__(self.name, model, param_names, **kwargs)

        self.loss = RootLoss(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
            faces=model.body_model.faces_tensor
        )

    def forward_pass(self, obs_data):
        pred_data = self.model.pred_params_mano(obs_data["is_right"])
        pred_data["cameras"] = self.model.params.get_cameras()

        vis_mask = obs_data["vis_mask"] >= 0
        loss, stats_dict = self.loss(obs_data, pred_data, vis_mask)
        return loss, stats_dict, pred_data


class SmoothOptimizer(StageOptimizer):
    name = "smooth_fit"
    stage = 1

    def __init__(
        self,
        model,
        all_loss_weights,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        param_names = ["trans", "root_orient", "betas", "latent_pose"]
        if model.opt_scale:
            param_names += ["world_scale"]
        if model.opt_cams:
            param_names += ["cam_f", "delta_cam_R"]

        super().__init__(self.name, model, param_names, **kwargs)

        self.loss = SMPLLoss(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
            faces=model.body_model.faces_tensor
        )

    def forward_pass(self, obs_data):
        pred_data = self.model.pred_params_mano(obs_data["is_right"])
        pred_data["cameras"] = self.model.params.get_cameras()
        pred_data.update(self.model.params.get_vars())
        pred_data["cam_R"], pred_data["cam_t"] = self.model.params.get_extrinsics()

        # compute data losses only
        vis_mask = obs_data["vis_mask"] >= 0
        loss, stats_dict = self.loss(obs_data, pred_data, self.model.seq_len, vis_mask)
        return loss, stats_dict, pred_data


