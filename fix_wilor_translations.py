"""
Fix WiLoR cam_trans scale for Dyn-HaMR compatibility.

WiLoR exports cam_trans for a virtual camera with focal_length ~ 37500 px,
but Dyn-HaMR uses the real camera intrinsics.
This script recomputes cam_trans so the MANO wrist projects to the
WiLoR-detected 2D keypoint under Dyn-HaMR's actual intrinsics.

Usage:
  python fix_wilor_translations.py --camera_dir cameras/ --tracks tracks/
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import torch

sys.path.insert(0, '.')
from body_model import MANO

parser = argparse.ArgumentParser()
parser.add_argument('--camera_dir', required=True, help='Path to directory containing cameras.npz')
parser.add_argument('--tracks', required=True, help='Path to tracks directory')
args = parser.parse_args()

# Look for cameras.json first, then cameras.npz, then any .npz in directory
cam_json = os.path.join(args.camera_dir, 'cameras.json')
cam_npz = os.path.join(args.camera_dir, 'cameras.npz')
if os.path.isfile(cam_json):
    with open(cam_json) as f:
        cam_data = json.load(f)
    fx, fy, cx, cy = cam_data['intrins']
    print(f"Loaded intrinsics from {cam_json}")
elif os.path.isfile(cam_npz):
    intr_files = [cam_npz]
    intr_data = np.load(intr_files[0], allow_pickle=True)
    if 'intrins' in intr_data:
        fx, fy, cx, cy = intr_data['intrins'][0]
    else:
        fx, fy, cx, cy = intr_data['data'][0]
    print(f"Loaded intrinsics from {cam_npz}")
else:
    intr_files = sorted(glob.glob(os.path.join(args.camera_dir, '*.npz')))
    if not intr_files:
        sys.exit(f"No cameras.json/.npz found in {args.camera_dir}")
    intr_data = np.load(intr_files[0], allow_pickle=True)
    if 'intrins' in intr_data:
        fx, fy, cx, cy = intr_data['intrins'][0]
    else:
        fx, fy, cx, cy = intr_data['data'][0]
    print(f"Loaded intrinsics from {intr_files[0]}")
print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

device = torch.device('cpu')
mano = MANO(batch_size=1, pose2rot=True, model_path='mano', is_rhand=True).to(device)

def get_root_loc(betas, global_orient, body_pose):
    """Get wrist joint location with trans=0."""
    with torch.no_grad():
        out = mano(
            betas=betas,
            global_orient=global_orient,
            hand_pose=body_pose,
            transl=torch.zeros(1, 3),
        )
    return out.joints[0, 0].cpu().numpy()


def estimate_cam_trans_from_keypoints(root_loc, wrist_2d, fx, fy, cx, cy, tz=1.0, is_right=1):
    """
    Solve for tx, ty that place the wrist at the 2D keypoint given tz.

    For right hands (is_right=1):
        wrist_cam = tx + root_loc  (MANO adds trans directly)
        x_2d = fx * (root_x + tx) / (root_z + tz) + cx
        => tx = (x_2d - cx) * (root_z + tz) / fx - root_x

    For left hands (is_right=0):
        Dyn-HaMR's run_mano() x-flips joints AFTER MANO forward pass.
        So wrist_cam[x] = -(tx + wrist_raw[x]) = -tx + root_loc[x]
        where root_loc is already x-flipped.
        => x_2d = fx * (-tx + root_x) / (root_z + tz) + cx
        => tx = root_x - (x_2d - cx) * (root_z + tz) / fx
    """
    x_2d, y_2d = wrist_2d
    if is_right:
        tx = (x_2d - cx) * (root_loc[2] + tz) / fx - root_loc[0]
    else:
        tx = root_loc[0] - (x_2d - cx) * (root_loc[2] + tz) / fx
    ty = (y_2d - cy) * (root_loc[2] + tz) / fy - root_loc[1]
    return np.array([tx, ty, tz], dtype=np.float32)


def estimate_tz_from_hand_span(keypoints, fx):
    """
    Estimate depth from the 2D hand span.
    WiLoR keypoints: 21 joints. Approx span from wrist to middle fingertip ~0.18m.
    """
    # Wrist is joint 0, middle fingertip is joint 12 (in OpenPose/MANO convention)
    kps = np.array(keypoints).reshape(-1, 3)
    if len(kps) < 13:
        return 1.0
    wrist = kps[0, :2]
    mid_tip = kps[12, :2]
    span_px = np.linalg.norm(wrist - mid_tip)
    if span_px < 5:
        return 1.0
    # MANO wrist-to-middle-tip distance is roughly 0.12-0.15m
    hand_m = 0.13
    tz = fx * hand_m / span_px
    # Clamp to reasonable hand depth
    tz = np.clip(tz, 0.3, 3.0)
    return float(tz)


def fix_track(track_dir):
    mano_files = sorted(glob.glob(os.path.join(track_dir, '*_mano.json')))
    fixed_count = 0
    for mano_path in mano_files:
        kp_path = mano_path.replace('_mano.json', '_keypoints.json')
        if not os.path.exists(kp_path):
            continue

        with open(mano_path, 'r') as f:
            mano_data = json.load(f)

        with open(kp_path, 'r') as f:
            kp_data = json.load(f)

        # Parse MANO params
        betas = torch.tensor(mano_data['betas'], dtype=torch.float32).unsqueeze(0)
        orient = torch.tensor(mano_data['global_orient'], dtype=torch.float32).unsqueeze(0)
        body_pose = torch.tensor(mano_data['body_pose'], dtype=torch.float32).reshape(1, 45)
        is_right = int(mano_data['is_right'])

        # Get root location
        root_loc = get_root_loc(betas, orient, body_pose)
        if is_right == 0:
            root_loc[0] *= -1  # Dyn-HaMR x-flips left hand vertices/joints

        # Parse keypoints
        people = kp_data.get('people', [])
        if len(people) == 0:
            continue
        kps = np.array(people[0]['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
        wrist_2d = kps[0, :2]

        # Estimate depth from hand span, fallback to 1.0m
        tz = estimate_tz_from_hand_span(kps, fx)

        # Compute corrected cam_trans
        new_trans = estimate_cam_trans_from_keypoints(root_loc, wrist_2d, fx, fy, cx, cy, tz=tz, is_right=is_right)

        old_trans = np.array(mano_data['cam_trans'], dtype=np.float32)
        mano_data['cam_trans'] = new_trans.tolist()

        with open(mano_path, 'w') as f:
            json.dump(mano_data, f, indent=2)

        fixed_count += 1
        if fixed_count <= 3 or fixed_count % 50 == 0:
            print(f"  {os.path.basename(mano_path)}: old={old_trans.round(3).tolist()}, new={new_trans.round(3).tolist()}, tz_est={tz:.2f}")

    print(f"Fixed {fixed_count} files in {track_dir}")


if __name__ == '__main__':
    for tid in sorted(os.listdir(args.tracks)):
        track_dir = os.path.join(args.tracks, tid)
        if not os.path.isdir(track_dir):
            continue
        print(f"\nProcessing track {tid}...")
        fix_track(track_dir)
