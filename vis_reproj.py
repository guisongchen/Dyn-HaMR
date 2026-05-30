#!/usr/bin/env python3
"""Visualize 3D-to-2D re-projection errors on original images.

Reads input (estimator) and optimized canonical JSON, computes 3D joints via
MANO, projects them to 2D, and overlays GT / input / optimized keypoints on
the source images with error lines and per-joint statistics.
"""
import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO

# ── MANO model (right-hand; left is handled by x-flip) ───────────
device = torch.device('cpu')
hand_model_r = MANO(batch_size=1, pose2rot=True, model_path='mano', is_rhand=True).to(device)

JNAMES = ['Wrist', 'T1', 'T2', 'T3', 'Ttip',
          'I1', 'I2', 'I3', 'Itip',
          'M1', 'M2', 'M3', 'Mtip',
          'R1', 'R2', 'R3', 'Rtip',
          'P1', 'P2', 'P3', 'Ptip']

HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

COLORS = {
    'gt': (0, 255, 0),
    'input': (0, 0, 255),
    'opt': (255, 128, 0),
    'err_input': (0, 0, 255),
    'err_opt': (255, 128, 0),
}


# ── Helpers ──────────────────────────────────────────────────────
def load_cameras(path):
    with open(path) as f:
        c = json.load(f)
    return {
        'w2c': np.array(c['w2c']),          # [T, 4, 4]
        'intrins': np.array(c['intrins']),  # [4]
        'height': c['height'],
        'width': c['width'],
        'num_frames': c['num_frames'],
    }


def load_hands(path):
    with open(path) as f:
        hands = json.load(f)['hands']
    result = []
    for h in hands:
        ir = h['is_right']
        frames = sorted(h['frames'], key=lambda f: f['frame_id'])
        T = max(f['frame_id'] for f in frames) + 1 if frames else 0
        j2d = np.zeros((T, 21, 3), dtype=np.float32)
        pose = np.zeros((T, 15, 3), dtype=np.float32)
        orient = np.zeros((T, 3), dtype=np.float32)
        trans = np.zeros((T, 3), dtype=np.float32)
        betas = None
        for f in frames:
            fid = f['frame_id']
            m = f['mano']
            k = f['keypoints']
            j2d[fid] = np.array(k['pose_keypoints_2d']).reshape(21, 3)
            pose[fid] = np.array(m['body_pose']).reshape(15, 3)
            orient[fid] = np.array(m['global_orient'])
            trans[fid] = np.array(m.get('world_trans', m.get('cam_trans')))
            if betas is None:
                betas = np.array(m['betas'])
        result.append({'is_right': ir, 'joints2d': j2d, 'pose': pose,
                       'orient': orient, 'trans': trans, 'betas': betas})
    return result


def run_mano(betas, poses, orients, trans, is_right_val):
    n = len(poses)
    j3d = np.zeros((n, 21, 3), dtype=np.float32)
    for i in range(n):
        with torch.no_grad():
            out = hand_model_r(
                betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                global_orient=torch.tensor(orients[i:i + 1], dtype=torch.float32),
                hand_pose=torch.tensor(poses[i:i + 1].reshape(1, 45), dtype=torch.float32),
                transl=torch.tensor(trans[i:i + 1], dtype=torch.float32),
            )
        j = out.joints.numpy()[0]
        if is_right_val == 0:
            j[:, 0] *= -1
        j3d[i] = j
    return j3d


def project(j3d, w2c_mat, intrins):
    fx, fy, cx, cy = intrins
    if w2c_mat is not None:
        R = w2c_mat[:, :3, :3]    # [T, 3, 3]
        t = w2c_mat[:, :3, 3]     # [T, 3]
        j_cam = np.einsum('tij,tkj->tki', R, j3d) + t[:, None, :]
    else:
        j_cam = j3d
    x = fx * j_cam[:, :, 0] / j_cam[:, :, 2] + cx
    y = fy * j_cam[:, :, 1] / j_cam[:, :, 2] + cy
    return np.stack([x, y], axis=-1)


def find_images(img_dir, num_frames):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    files = sorted(files)
    if len(files) >= num_frames:
        return files[:num_frames]
    # Fallback: try zero-padded names
    files2 = []
    for t in range(num_frames):
        for name in (f'{t:06d}.jpg', f'{t:06d}.png', f'{t}.jpg', f'{t}.png',
                     f'frame_{t:06d}.jpg', f'img_{t:06d}.jpg'):
            p = os.path.join(img_dir, name)
            if os.path.exists(p):
                files2.append(p)
                break
    if len(files2) == num_frames:
        return files2
    return files  # best effort


def draw_hand_skeleton(img, kpts2d, color, mask=None, thickness=2, radius=4):
    for u, v in HAND_SKELETON:
        if mask is not None and (not mask[u] or not mask[v]):
            continue
        pt1 = tuple(kpts2d[u].astype(int))
        pt2 = tuple(kpts2d[v].astype(int))
        if all(c > 0 for c in pt1 + pt2):
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for j, pt in enumerate(kpts2d):
        if mask is not None and not mask[j]:
            continue
        p = tuple(pt.astype(int))
        if all(c > 0 for c in p):
            cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_error_line(img, pt_gt, pt_pred, color, thickness=1):
    p1 = tuple(pt_gt.astype(int))
    p2 = tuple(pt_pred.astype(int))
    if all(c > 0 for c in p1 + p2):
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def render_frame(img_bgr, gt_kpts, in_kpts, opt_kpts, mask, hand_label,
                 mean_err_in, mean_err_opt):
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    # Only draw skeleton for optimized (cleanest)
    draw_hand_skeleton(canvas, opt_kpts, COLORS['opt'], mask=mask, thickness=2, radius=5)

    # GT keypoints
    for j in range(21):
        if not mask[j]:
            continue
        pt = tuple(gt_kpts[j].astype(int))
        if all(c > 0 for c in pt):
            cv2.circle(canvas, pt, 3, COLORS['gt'], -1, cv2.LINE_AA)

    # Error lines
    for j in range(21):
        if not mask[j]:
            continue
        draw_error_line(canvas, gt_kpts[j], in_kpts[j], COLORS['err_input'], 1)
        draw_error_line(canvas, gt_kpts[j], opt_kpts[j], COLORS['err_opt'], 1)

    # Text overlay
    y0 = 30
    dy = 28
    texts = [
        f'{hand_label}  GT(green)  In(red)  Opt(orange)',
        f'Mean error:  in={mean_err_in:.1f}px  opt={mean_err_opt:.1f}px',
    ]
    for i, txt in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(canvas, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Per-joint error legend (top-right)
    errs = np.sqrt(np.sum((opt_kpts - gt_kpts) ** 2, axis=-1))
    worst = np.argsort(-errs)[:5]
    x0 = w - 280
    y0 = 30
    for rank, j in enumerate(worst):
        if not mask[j]:
            continue
        txt = f'{JNAMES[j]}: {errs[j]:.1f}px'
        y = y0 + rank * 22
        cv2.putText(canvas, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLORS['opt'], 1, cv2.LINE_AA)

    return canvas


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize 3D-2D reprojection errors')
    parser.add_argument('--input_hands', default='demo_data/hands.json')
    parser.add_argument('--opt_hands', default='outputs/hands.json')
    parser.add_argument('--opt_cameras', default='outputs/cameras.json')
    parser.add_argument('--img_dir', default=None,
                        help='Source images directory (default: <input_hands_dir>/frames)')
    parser.add_argument('--out_dir', default='outputs/reproj_vis')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Limit number of frames to process')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS if writing mp4')
    parser.add_argument('--write_video', action='store_true',
                        help='Also write an MP4 video')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Auto-detect image directory
    if args.img_dir is None:
        args.img_dir = os.path.join(os.path.dirname(args.input_hands), 'frames')

    in_hands = load_hands(args.input_hands)
    opt_hands = load_hands(args.opt_hands)
    opt_cams = load_cameras(args.opt_cameras)
    T = opt_cams['num_frames']
    B = len(opt_hands)

    # Compute 3D joints
    in_joints3d, opt_joints3d = [], []
    for h in range(B):
        in_joints3d.append(run_mano(in_hands[h]['betas'], in_hands[h]['pose'],
                                     in_hands[h]['orient'], in_hands[h]['trans'],
                                     in_hands[h]['is_right']))
        opt_joints3d.append(run_mano(opt_hands[h]['betas'], opt_hands[h]['pose'],
                                      opt_hands[h]['orient'], opt_hands[h]['trans'],
                                      opt_hands[h]['is_right']))

    # Project to 2D
    opt_joints2d = [project(j, opt_cams['w2c'], opt_cams['intrins']) for j in opt_joints3d]
    in_joints2d_proj = [project(j, None, opt_cams['intrins']) for j in in_joints3d]

    # Load images
    img_paths = find_images(args.img_dir, T)
    if len(img_paths) < T:
        print(f'Warning: found {len(img_paths)} images for {T} frames')

    frame_ids = list(range(min(T, len(img_paths)) if img_paths else T))
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]

    HNAMES = ['Left', 'Right']
    video_writers = [None] * B

    for t in frame_ids:
        if img_paths and os.path.exists(img_paths[t]):
            img = cv2.imread(img_paths[t])
        else:
            img = np.zeros((opt_cams['height'], opt_cams['width'], 3), dtype=np.uint8)

        for h in range(B):
            ih = in_hands[h]
            mask = ih['joints2d'][t, :, 2] > 0.3

            gt_k = ih['joints2d'][t, :, :2]
            in_k = in_joints2d_proj[h][t]
            opt_k = opt_joints2d[h][t]

            e_in = np.sqrt(np.sum((in_k - gt_k) ** 2, axis=-1))
            e_opt = np.sqrt(np.sum((opt_k - gt_k) ** 2, axis=-1))
            mean_in = e_in[mask].mean() if mask.sum() > 0 else 0.0
            mean_opt = e_opt[mask].mean() if mask.sum() > 0 else 0.0

            canvas = render_frame(img, gt_k, in_k, opt_k, mask,
                                  HNAMES[h], mean_in, mean_opt)

            out_path = os.path.join(frames_dir, f'hand{h}_{t:06d}.png')
            cv2.imwrite(out_path, canvas)

            if args.write_video:
                if video_writers[h] is None:
                    vpath = os.path.join(args.out_dir, f'reproj_{HNAMES[h].lower()}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers[h] = cv2.VideoWriter(vpath, fourcc, args.fps,
                                                        (canvas.shape[1], canvas.shape[0]))
                video_writers[h].write(canvas)

        if t % 30 == 0 or t == frame_ids[-1]:
            print(f'Processed frame {t}/{len(frame_ids)}')

    for vw in video_writers:
        if vw is not None:
            vw.release()

    print(f'Done. Frames saved to {frames_dir}')
    if args.write_video:
        print(f'Videos saved to {args.out_dir}')


if __name__ == '__main__':
    main()
