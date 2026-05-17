import os
import numpy as np

import torch
from tqdm import tqdm

from body_model import OP_IGNORE_JOINTS
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
        vis_every=-1,
        **kwargs,
    ):
        self.name = name
        self.model = model

        self.set_opt_vars(param_names)

        self.optim = torch.optim.LBFGS(
            self.opt_params, max_iter=lbfgs_max_iter, lr=lr, line_search_fn=LINE_SEARCH
        )

        self.vis_every = vis_every

        self.cur_step = 0
        self.cur_loss = 0

    def set_opt_vars(self, param_names):

        self.param_names = param_names
        self.model.params.set_require_grads(self.param_names)
        self.opt_params = [
            getattr(self.model.params, name) for name in self.param_names
        ]

    def forward_pass(self, obs_data):
        raise NotImplementedError

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
            np.savez(out_path, **results)

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

    def run(self, obs_data, num_iters, out_dir, vis=None):
        self.cur_step = 0
        res_dir = os.path.join(out_dir, self.name)
        os.makedirs(res_dir, exist_ok=True)
        seq_name = obs_data["seq_name"][0]


        self._obs_data = obs_data

        def closure():
            self.optim.zero_grad()
            loss, _, _ = self.forward_pass(self._obs_data)
            self.cur_loss = loss.detach().cpu().item()
            loss.backward()
            return loss

        for i in tqdm(range(num_iters), desc=self.name, leave=False):
            self.optim.step(closure)

            if np.isnan(self.cur_loss):
                raise ValueError

        self.cur_step = num_iters
        self.save_results(res_dir, seq_name)
        self.vis_result(res_dir, obs_data, vis)


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


