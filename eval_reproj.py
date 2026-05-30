#!/usr/bin/env python3
"""Compute and visualize 3D-to-2D re-projection errors for optimized results.

Reads cameras.json + hands.json (with pre-computed 3D keypoints).
No dependency on the original input hands.json.
Outputs two dense figures: temporal analysis + joint distribution.
"""
import os
import sys
import json
import argparse

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, os.path.dirname(__file__))

# ── MANO fallback (only used if 3D keypoints are missing) ──────
hand_model_r = None


def get_mano_model():
    global hand_model_r
    if hand_model_r is None:
        import torch
        from body_model import MANO
        hand_model_r = MANO(
            batch_size=1, pose2rot=True, model_path='mano', is_rhand=True
        ).to('cpu')
    return hand_model_r


def run_mano_frame(betas, pose, orient, trans, is_right_val):
    import torch
    model = get_mano_model()
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
            global_orient=torch.tensor(orient[None], dtype=torch.float32),
            hand_pose=torch.tensor(pose[None].reshape(1, 45), dtype=torch.float32),
            transl=torch.tensor(trans[None], dtype=torch.float32),
        )
    j = out.joints.numpy()[0]
    if is_right_val == 0:
        j[:, 0] *= -1
    return j


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
    has_3d = True
    for h in hands:
        ir = h['is_right']
        frames = sorted(h['frames'], key=lambda f: f['frame_id'])
        T = max(f['frame_id'] for f in frames) + 1 if frames else 0
        j2d = np.zeros((T, 21, 3), dtype=np.float32)
        j3d = np.zeros((T, 21, 3), dtype=np.float32)
        mano_params = []
        for f in frames:
            fid = f['frame_id']
            k = f['keypoints']
            m = f['mano']
            j2d[fid] = np.array(k['pose_keypoints_2d']).reshape(21, 3)
            if 'pose_keypoints_3d' in k:
                j3d[fid] = np.array(k['pose_keypoints_3d']).reshape(21, 3)
            else:
                has_3d = False
                mano_params.append((fid, m, ir))
        result.append({
            'is_right': ir,
            'joints2d': j2d,
            'joints3d': j3d,
            'mano_params': mano_params,
        })
    return result, has_3d


def project(j3d, w2c_mat, intrins):
    fx, fy, cx, cy = intrins
    if w2c_mat is not None:
        R = w2c_mat[:, :3, :3]
        t = w2c_mat[:, :3, 3]
        j_cam = np.einsum('tij,tkj->tki', R, j3d) + t[:, None, :]
    else:
        j_cam = j3d
    x = fx * j_cam[:, :, 0] / j_cam[:, :, 2] + cx
    y = fy * j_cam[:, :, 1] / j_cam[:, :, 2] + cy
    return np.stack([x, y], axis=-1)


def hand_label(is_right):
    return 'Right' if is_right else 'Left'


def hand_color(is_right):
    return '#FF5722' if is_right else '#2196F3'


# ── Visualization ────────────────────────────────────────────────
JNAMES = ['Wrist', 'T1', 'T2', 'T3', 'Ttip',
          'I1', 'I2', 'I3', 'Itip',
          'M1', 'M2', 'M3', 'Mtip',
          'R1', 'R2', 'R3', 'Rtip',
          'P1', 'P2', 'P3', 'Ptip']


def plot_summary(all_errs, out_dir):
    """Single 2xB figure: row 0 = temporal, row 1 = boxplot."""
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    B = len(all_errs)
    fig, axes = plt.subplots(2, B, figsize=(7 * B, 8.5), sharey='row',
                             squeeze=False, constrained_layout=True)

    for h in range(B):
        e = all_errs[h]['e']
        mask = all_errs[h]['mask']
        label = all_errs[h]['label']
        color = all_errs[h]['color']
        T = e.shape[0]

        # ── Row 0: temporal ──
        ax = axes[0, h]
        for j in range(21):
            e_j = np.where(mask[:, j], e[:, j], np.nan)
            ax.plot(e_j, alpha=0.18, lw=0.4, color='#555555')

        frame_mean = np.array([
            e[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan
            for t in range(T)
        ])
        ax.plot(frame_mean, color=color, lw=1.4, label='mean')
        ax.fill_between(range(T), 0, frame_mean, alpha=0.12, color=color)

        ax.axhline(y=5, color='gray', ls='--', alpha=0.3, lw=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('px')
        ax.set_title(f'{label} — reprojection error')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=7)

        # ── Row 1: boxplot ──
        ax = axes[1, h]
        joint_data = [e[:, j][mask[:, j]] for j in range(21)]
        bp = ax.boxplot(joint_data, patch_artist=True, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 1})
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        ax.set_xticklabels(JNAMES, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('px')
        ax.set_title(f'{label} — per-joint distribution')
        ax.grid(True, alpha=0.3, axis='y')
        ymax = max(np.percentile(d, 95) if len(d) > 0 else 0 for d in joint_data)
        ax.set_ylim(0, max(ymax * 1.2, 2))

    p = os.path.join(out_dir, 'reproj_summary.png')
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f'Saved {p}')


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D-2D reprojection errors')
    parser.add_argument('--hands', default='outputs/hands.json')
    parser.add_argument('--cameras', default='outputs/cameras.json')
    parser.add_argument('--out_dir', default='outputs/reproj_eval')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cams = load_cameras(args.cameras)
    hands, has_3d = load_hands(args.hands)
    T = cams['num_frames']
    B = len(hands)

    # Fallback: compute 3D joints via MANO if not present in JSON
    if not has_3d:
        print('Warning: hands.json missing pose_keypoints_3d, computing via MANO...')
        for h in range(B):
            for fid, m, ir in hands[h]['mano_params']:
                j3d = run_mano_frame(
                    np.array(m['betas']),
                    np.array(m['body_pose']).reshape(15, 3),
                    np.array(m['global_orient']),
                    np.array(m['world_trans']),
                    ir,
                )
                hands[h]['joints3d'][fid] = j3d

    # Project 3D -> 2D and compute errors
    all_errs = []
    print('\n' + '=' * 60)
    print('REPROJECTION ERROR REPORT')
    print('=' * 60)
    for h in range(B):
        j3d = hands[h]['joints3d'][:T]
        j2d_gt = hands[h]['joints2d'][:T][:, :, :2]
        mask = hands[h]['joints2d'][:T][:, :, 2] > 0.3

        j2d_pred = project(j3d, cams['w2c'], cams['intrins'])
        e = np.sqrt(np.sum((j2d_pred - j2d_gt) ** 2, axis=-1))

        label = hand_label(hands[h]['is_right'])
        color = hand_color(hands[h]['is_right'])
        e_valid = e[mask]
        print(f'\n{label}:')
        print(f'  mean={e_valid.mean():.2f}px  median={np.median(e_valid):.2f}px  '
              f'p95={np.percentile(e_valid, 95):.2f}px  max={e_valid.max():.2f}px')

        joint_means = [
            e[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else np.nan
            for j in range(21)
        ]
        worst = np.argsort([-x if not np.isnan(x) else -1 for x in joint_means])[:5]
        print(f'  Worst joints: {", ".join(f"{JNAMES[j]}={joint_means[j]:.1f}" for j in worst)}')

        all_errs.append({'e': e, 'mask': mask, 'label': label, 'color': color})

    # Single dense summary figure
    plot_summary(all_errs, args.out_dir)

    print(f'\nDone. Results in {args.out_dir}')


if __name__ == '__main__':
    main()
