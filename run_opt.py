import os
import sys
import random
import numpy as np
import time

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from data import get_dataset_from_cfg

from optim.base_scene import BaseSceneModel

from optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer
)
from optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from vis.viewer import init_viewer
from body_model import MANO
from util.tensor import get_device, move_to

from run_vis import run_vis

sys.path.append(os.path.join(os.path.dirname(__file__), 'HMP'))

N_STAGES = 3

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_opt(cfg, dataset, out_dir, device):
    a = time.time()
    args = cfg.data
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    obs_data = move_to(next(iter(loader)), device)
    cam_data = move_to(dataset.get_camera_data(), device)

    cam_R, cam_t = dataset.cam_data.cam2world()
    intrins = dataset.cam_data.intrins
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins)

    # check whether the cameras are static
    # if static, cannot optimize scale
    cfg.model.opt_scale &= not dataset.cam_data.is_static

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]

    cfg.paths.base_dir = os.path.abspath(os.path.dirname(__file__))
    cfg.paths.MANO_DIR = os.path.join(cfg.paths.base_dir, "mano")
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    hand_model = MANO(batch_size=B * T, pose2rot=True, **mano_cfg).to(device)

    ################################################################
    ######################## optimization ##########################
    ################################################################
    margs = cfg.model
    base_model = BaseSceneModel(
        B, T, hand_model, None, **margs
    )

    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_input_poses(dataset, os.path.join(out_dir, "hamer"), args.seq)
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)

    opts = cfg.optim.options
    vis_scale = 0.25
    vis = None
    if opts.vis_every > 0:
        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
            fps=cfg.fps,
        )
    print("OPTIMIZER OPTIONS:", opts)

    a = time.time()
    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis)

    args = cfg.optim.smooth
    b = time.time()
    print('root optimization time: ', b - a)
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data, args.num_iters, out_dir, vis)
    c = time.time()
    print('Smooth optimization time: ', c - b)

    # HMP
    prior_out = os.path.join(out_dir, 'prior')
    has_prior_results = os.path.isdir(prior_out) and any(
        f.endswith('_world_results.npz') for f in os.listdir(prior_out)
    )
    if cfg.run_prior and not has_prior_results:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        from HMP.fitting import run_prior
        run_prior(cfg, dataset, out_dir, device, ['smooth_fit'], \
        obs_data, hand_model, cfg, cfg.data, prior_out)
    d = time.time()
    print('prior optimization time: ', d-c)


def load_config():
    base = os.path.dirname(__file__)
    cfg = OmegaConf.load(os.path.join(base, "confs/config.yaml"))
    data_cfg = OmegaConf.load(os.path.join(base, "confs/data/demo_dynhamr.yaml"))
    optim_cfg = OmegaConf.load(os.path.join(base, "confs/optim.yaml"))
    return OmegaConf.merge({"data": data_cfg}, optim_cfg, cfg)


def main():
    cfg = load_config()

    set_seed(cfg.get('seed', 42))

    out_dir = os.path.abspath("outputs")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    print("out_dir", out_dir)

    dataset = get_dataset_from_cfg(cfg)
    save_track_info(dataset, out_dir)

    if cfg.run_opt:
        device = get_device()
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        run_vis(cfg, dataset, out_dir, 0, **cfg.get("vis", dict()))


if __name__ == "__main__":
    main()
