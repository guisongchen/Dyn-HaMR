import os
import sys
import random
import json
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
from vis.viewer import init_viewer
from body_model import MANO
from util.tensor import get_device, move_to

from run_vis import run_vis
from body_model.utils import run_mano

sys.path.append(os.path.join(os.path.dirname(__file__), 'HMP'))

N_STAGES = 3

def set_seed(seed=42):
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

    cfg.model.opt_scale &= not dataset.cam_data.is_static

    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]

    cfg.paths.base_dir = os.path.abspath(os.path.dirname(__file__))
    cfg.paths.MANO_DIR = os.path.join(cfg.paths.base_dir, "mano")
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    hand_model = MANO(batch_size=B * T, pose2rot=True, **mano_cfg).to(device)

    margs = cfg.model
    base_model = BaseSceneModel(
        B, T, hand_model, None, **margs
    )

    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

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
    optim = RootOptimizer(base_model, stage_loss_weights, save_results=False, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis)

    b = time.time()
    print('root optimization time: ', b - a)
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=cfg.optim.smooth.opt_scale, **opts
    )
    optim.run(obs_data, cfg.optim.smooth.num_iters, out_dir, vis)
    c = time.time()
    print('Smooth optimization time: ', c - b)

    export_canonical_json(
        os.path.join(out_dir, "smooth_fit", f"{args.seq}_000300_world_results.npz"),
        os.path.join(cfg.data.root, "hands.json"),
        out_dir,
        hand_model,
    )

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


def export_canonical_json(opt_npz_path, hands_json_path, out_dir, body_model):
    """Convert optimized .npz to canonical cameras.json + hands.json format."""
    opt = np.load(opt_npz_path)
    with open(hands_json_path) as f:
        in_hands = json.load(f)['hands']

    B, T = opt['pose_body'].shape[:2]

    # --- cameras.json ---
    cam_R = opt['cam_R'][0]  # [T, 3, 3] W2C
    cam_t = opt['cam_t'][0]  # [T, 3]
    w2c = np.zeros((T, 4, 4))
    w2c[:, :3, :3] = cam_R
    w2c[:, :3, 3] = cam_t
    w2c[:, 3, 3] = 1.0

    cameras = {
        "intrins": opt['intrins'].tolist(),
        "height": 1080,
        "width": 1920,
        "num_frames": T,
        "w2c": w2c.tolist(),
    }
    cam_out = os.path.join(out_dir, "cameras.json")
    with open(cam_out, "w") as f:
        json.dump(cameras, f, indent=1)
    print(f"Exported {cam_out}")

    # --- compute 3D joints via MANO ---
    device = next(body_model.buffers()).device
    trans_t = torch.from_numpy(opt['trans']).float().to(device)          # [B, T, 3]
    root_orient_t = torch.from_numpy(opt['root_orient']).float().to(device)  # [B, T, 3]
    body_pose_t = torch.from_numpy(opt['pose_body']).float().to(device).reshape(B, T, -1)  # [B, T, 45]
    is_right_t = torch.from_numpy(opt['is_right']).float().to(device)    # [B, T] or [B, 1]
    betas_t = torch.from_numpy(opt['betas']).float().to(device)          # [B, 10]
    with torch.no_grad():
        mano_out = run_mano(body_model, trans_t, root_orient_t, body_pose_t, is_right_t, betas_t)
        joints3d = mano_out["joints"].cpu().numpy()  # [B, T, J, 3]

    # --- hands.json ---
    hands_out = []
    for h in range(B):
        ir = int(opt['is_right'][h, 0])
        poses = opt['pose_body'][h]       # [T, 15, 3]
        orients = opt['root_orient'][h]   # [T, 3]
        trans = opt['trans'][h]           # [T, 3]  world space
        betas = opt['betas'][h]           # [10]

        in_frames = {f['frame_id']: f for f in in_hands[h]['frames']}

        frames = []
        for t in range(T):
            in_f = in_frames.get(t, {})
            kp = in_f.get('keypoints', {}).get('pose_keypoints_2d', [0] * 63)

            frames.append({
                "frame_id": t,
                "mano": {
                    "betas": betas.tolist(),
                    "body_pose": poses[t].reshape(45).tolist(),
                    "global_orient": orients[t].tolist(),
                    "world_trans": trans[t].tolist(),
                },
                "keypoints": {
                    "pose_keypoints_2d": kp,
                    "pose_keypoints_3d": joints3d[h, t].reshape(-1).tolist(),
                }
            })

        hands_out.append({
            "is_right": ir,
            "frames": frames,
        })

    hands_out_path = os.path.join(out_dir, "hands.json")
    with open(hands_out_path, "w") as f:
        json.dump({"hands": hands_out}, f, indent=1)
    print(f"Exported {hands_out_path}")


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
    print("out_dir", out_dir)

    dataset = get_dataset_from_cfg(cfg)

    if cfg.run_opt:
        os.chdir(out_dir)
        device = get_device()
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        run_vis(cfg, dataset, out_dir, 0, **cfg.get("vis", dict()))


if __name__ == "__main__":
    main()
