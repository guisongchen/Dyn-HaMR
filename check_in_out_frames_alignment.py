"""Verify frame-to-frame alignment between cameras, hands, and 2D keypoints.

This script checks:
1. Input 2D keypoints are preserved in output hands.json (they should be identical).
2. Input MANO params in camera-space reproject well to 2D keypoints (intrinsics only).
3. World-space initialization from dataset reprojects correctly using full cameras.
4. Camera and hand frame counts match after dataset alignment.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO
from body_model.utils import run_mano
from data import get_dataset_from_cfg
from util.tensor import get_device, move_to
from omegaconf import OmegaConf


def project(j3d, w2c_mat, intrins):
    """Project 3D joints to 2D. j3d: [T, J, 3], w2c_mat: [T, 4, 4] or None, intrins: [4] or [T, 4]."""
    fx, fy, cx, cy = intrins[0] if intrins.ndim == 2 else intrins
    if w2c_mat is not None:
        R = w2c_mat[:, :3, :3]
        t = w2c_mat[:, :3, 3]
        j_cam = np.einsum('tij,tkj->tki', R, j3d) + t[:, None, :]
    else:
        j_cam = j3d
    x = fx * j_cam[:, :, 0] / j_cam[:, :, 2] + cx
    y = fy * j_cam[:, :, 1] / j_cam[:, :, 2] + cy
    return np.stack([x, y], axis=-1)


def compute_reprojection_error(j2d_pred, j2d_gt, mask):
    e = np.sqrt(np.sum((j2d_pred - j2d_gt) ** 2, axis=-1))
    e_valid = e[mask]
    if len(e_valid) == 0:
        return np.nan, np.nan, np.nan
    return e_valid.mean(), np.median(e_valid), e_valid.max()


def check_keypoint_preservation(input_hands_path, output_hands_path):
    """Check that output hands.json preserves input 2D keypoints."""
    with open(input_hands_path) as f:
        in_data = json.load(f)
    with open(output_hands_path) as f:
        out_data = json.load(f)

    print("\n=== Keypoint Preservation Check ===")
    mismatches = 0
    total_frames = 0
    for h, (in_hand, out_hand) in enumerate(zip(in_data['hands'], out_data['hands'])):
        in_frames = {f['frame_id']: f for f in in_hand['frames']}
        out_frames = {f['frame_id']: f for f in out_hand['frames']}
        common_fids = sorted(set(in_frames.keys()) & set(out_frames.keys()))

        for fid in common_fids:
            in_kp = np.array(in_frames[fid]['keypoints']['pose_keypoints_2d'])
            out_kp = np.array(out_frames[fid]['keypoints']['pose_keypoints_2d'])
            total_frames += 1
            if not np.allclose(in_kp, out_kp, atol=1e-4):
                mismatches += 1
                diff = np.abs(in_kp - out_kp).max()
                if mismatches <= 3:
                    print(f"  Hand {h}, frame {fid}: keypoints differ (max diff={diff:.4f})")

    print(f"  Common frames checked: {total_frames}, mismatches: {mismatches}")
    if mismatches == 0:
        print("  PASS: Output preserves all input 2D keypoints.")
    else:
        print(f"  FAIL: {mismatches}/{total_frames} frames have different keypoints.")
    return mismatches == 0


def check_dataset_alignment(cfg):
    """Check that dataset loads cameras and hands with correct alignment."""
    print("\n=== Dataset Alignment Check ===")
    device = get_device()
    dataset = get_dataset_from_cfg(cfg)
    dataset.load_data(interp_input=False)

    print(f"  Dataset: B={len(dataset)}, seq_len={dataset.seq_len}")
    print(f"  data_start={dataset.data_start}, data_end={dataset.data_end}")

    cam_data = dataset.get_camera_data()
    print(f"  Camera: cam_R shape={cam_data['cam_R'].shape}, cam_t shape={cam_data['cam_t'].shape}")

    # Load MANO model
    cfg.paths.base_dir = os.path.abspath(os.path.dirname(__file__))
    cfg.paths.MANO_DIR = os.path.join(cfg.paths.base_dir, "mano")
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    hand_model = MANO(batch_size=len(dataset) * dataset.seq_len, pose2rot=True, **mano_cfg).to(device)

    errors_cam = []
    errors_world = []

    for idx in range(len(dataset)):
        obs_data = dataset[idx]
        B_i, T_i = 1, dataset.seq_len

        # Run MANO with init params (camera-space trans)
        trans = obs_data['init_trans'][None]  # [1, T, 3]
        root_orient = obs_data['init_root_orient'][None]
        body_pose = obs_data['init_body_pose'][None]
        is_right = obs_data['is_right'][None]
        betas = obs_data['init_body_shape'][None].mean(1, keepdim=True)

        with torch.no_grad():
            mano_out = run_mano(hand_model, trans, root_orient, body_pose, is_right, betas)
            j3d_cam = mano_out['joints'].cpu().numpy()[0]  # [T, 21, 3]

        # Project with intrinsics only (camera space)
        intrins = cam_data['intrins'][0].cpu().numpy()  # [4]
        j2d_proj_cam = project(j3d_cam, None, intrins)

        # Project with full cameras (world space conversion)
        cam_R = cam_data['cam_R'].cpu().numpy()
        cam_t = cam_data['cam_t'].cpu().numpy()
        w2c = np.zeros((T_i, 4, 4))
        w2c[:, :3, :3] = cam_R
        w2c[:, :3, 3] = cam_t
        j2d_proj_world = project(j3d_cam, w2c, intrins)

        j2d_gt = obs_data['joints2d'].cpu().numpy()  # [T, 21, 3]
        mask = j2d_gt[:, :, 2] > 0.3

        err_cam = compute_reprojection_error(j2d_proj_cam, j2d_gt[:, :, :2], mask)
        err_world = compute_reprojection_error(j2d_proj_world, j2d_gt[:, :, :2], mask)

        errors_cam.append(err_cam)
        errors_world.append(err_world)

        label = 'Right' if int(obs_data['is_right'][0]) == 1 else 'Left'
        print(f"  Hand {idx} ({label}):")
        print(f"    Camera-space reprojection (intrinsics only):")
        print(f"      mean={err_cam[0]:.2f}px  median={err_cam[1]:.2f}px  max={err_cam[2]:.2f}px")
        print(f"    World-space reprojection (full camera + W2C):")
        print(f"      mean={err_world[0]:.2f}px  median={err_world[1]:.2f}px  max={err_world[2]:.2f}px")

    print("\n=== Interpretation ===")
    for idx, (err_cam, err_world) in enumerate(zip(errors_cam, errors_world)):
        if np.isnan(err_cam[0]):
            continue
        if err_cam[0] < 10:
            print(f"  Hand {idx}: Input MANO params match 2D keypoints well (camera-space).")
        else:
            print(f"  Hand {idx}: Input MANO params DO NOT match 2D keypoints (err={err_cam[0]:.1f}px).")
            print("           This means the tracker's 3D estimate is inconsistent with its 2D detections.")

        if err_world[0] < 10:
            print(f"  Hand {idx}: Camera trajectory + input pose = good 2D fit.")
        elif err_cam[0] < 10 and err_world[0] > 10:
            print(f"  Hand {idx}: Camera trajectory is BAD (camera-space good, world-space bad).")
            print("           The W2C transform does not map the camera-space pose back to the image.")
        else:
            print(f"  Hand {idx}: Both camera trajectory and input pose may have issues.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hands', default='demo_data/hands.json')
    parser.add_argument('--output_hands', default='outputs/hands.json')
    parser.add_argument('--run_dataset_check', action='store_true',
                        help='Also load the dataset and check camera/hand alignment')
    args = parser.parse_args()

    if os.path.isfile(args.input_hands) and os.path.isfile(args.output_hands):
        check_keypoint_preservation(args.input_hands, args.output_hands)
    else:
        print(f"Skipping keypoint check (missing {args.input_hands} or {args.output_hands})")

    if args.run_dataset_check:
        cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "confs/config.yaml"))
        data_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "confs/data/demo_dynhamr.yaml"))
        optim_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "confs/optim.yaml"))
        cfg = OmegaConf.merge({"data": data_cfg}, optim_cfg, cfg)
        check_dataset_alignment(cfg)


if __name__ == '__main__':
    main()
