"""Diagnose optimization quality — find where input data degrades and what the optimizer fixes.

No ground truth exists. This script surfaces inconsistencies between the estimator's
per-frame predictions and the temporally-smoothed optimization output.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO

parser = argparse.ArgumentParser()
parser.add_argument('--opt_npz', default='outputs/smooth_fit/dynhamr_out_000300_world_results.npz')
parser.add_argument('--hands_json', default='demo_data/hands.json')
parser.add_argument('--cameras_json', default='demo_data/cameras.json')
parser.add_argument('--out_dir', default='outputs/eval')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────
opt = np.load(args.opt_npz)
with open(args.hands_json) as f:
    hands = json.load(f)['hands']
with open(args.cameras_json) as f:
    cams_data = json.load(f)

device = torch.device('cpu')
hand_model_r = MANO(batch_size=1, pose2rot=True, model_path='mano', is_rhand=True).to(device)

B, T = opt['pose_body'].shape[:2]
fx, fy, cx, cy = opt['intrins'].astype(np.float64)
cam_R = opt['cam_R'][0]
cam_t = opt['cam_t'][0]

# Load per-frame estimator data
in_joints2d, in_poses, in_orients, in_trans, in_betas_arr = [], [], [], [], []
in_frame_ids = []
for h in hands:
    frames = h['frames']
    j2d = np.zeros((T, 21, 3), dtype=np.float32)
    pose = np.zeros((T, 15, 3), dtype=np.float32)
    orient = np.zeros((T, 3), dtype=np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    fids = set()
    for f in frames:
        fid = f['frame_id']
        if fid < T:
            j2d[fid] = np.array(f['keypoints']['pose_keypoints_2d']).reshape(21, 3)
            pose[fid] = np.array(f['mano']['body_pose']).reshape(15, 3)
            orient[fid] = np.array(f['mano']['global_orient'])
            trans[fid] = np.array(f['mano']['cam_trans'])
            fids.add(fid)
    in_joints2d.append(j2d)
    in_poses.append(pose)
    in_orients.append(orient)
    in_trans.append(trans)
    in_betas_arr.append(np.array(frames[0]['mano']['betas']))
    in_frame_ids.append(sorted(fids))


# ── MANO forward ─────────────────────────────────────────────────
def run_mano(betas, poses, orients, trans, is_right_val):
    n = len(betas)
    j3d = np.zeros((n, 21, 3), dtype=np.float32)
    for i in range(n):
        with torch.no_grad():
            out = hand_model_r(
                betas=torch.tensor(betas[i:i+1], dtype=torch.float32),
                global_orient=torch.tensor(orients[i:i+1], dtype=torch.float32),
                hand_pose=torch.tensor(poses[i:i+1].reshape(1, 45), dtype=torch.float32),
                transl=torch.tensor(trans[i:i+1], dtype=torch.float32),
            )
        j = out.joints.numpy()[0]
        if is_right_val == 0:
            j[:, 0] *= -1
        j3d[i] = j
    return j3d

opt_joints3d, in_joints3d = [], []
for h in range(B):
    ir = int(opt['is_right'][h, 0])
    opt_joints3d.append(run_mano(np.tile(opt['betas'][h], (T, 1)), opt['pose_body'][h],
                                  opt['root_orient'][h], opt['trans'][h], ir))
    in_joints3d.append(run_mano(np.tile(in_betas_arr[h], (T, 1)), in_poses[h],
                                 in_orients[h], in_trans[h], ir))


# ── Projection ───────────────────────────────────────────────────
def project(j3d, w2c=False):
    if w2c:
        j_cam = np.einsum('tij,tkj->tki', cam_R, j3d) + cam_t[:, None, :]
    else:
        j_cam = j3d
    x = fx * j_cam[:, :, 0] / j_cam[:, :, 2] + cx
    y = fy * j_cam[:, :, 1] / j_cam[:, :, 2] + cy
    return np.stack([x, y], axis=-1)

opt_joints2d = [project(j, w2c=True) for j in opt_joints3d]
in_joints2d_proj = [project(j, w2c=False) for j in in_joints3d]


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

JNAMES = ['Wrist','I1','I2','I3','M1','M2','M3','P1','P2','P3',
          'R1','R2','R3','T1','T2','T3','Tp1','Tp2','Tp3','Tp4','Tp5']
HNAMES = ['Left', 'Right']

print('=' * 70)
print('DIAGNOSTIC REPORT')
print('=' * 70)

for h in range(B):
    label = HNAMES[h]
    fids = in_frame_ids[h]
    e_opt = np.sqrt(np.sum((opt_joints2d[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    e_in = np.sqrt(np.sum((in_joints2d_proj[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    mask = in_joints2d[h][:, :, 2] > 0.3

    opt_per_frame = np.array([e_opt[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    in_per_frame = np.array([e_in[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])

    print(f'\n─ Hand: {label} ({len(fids)} frames with data, f{min(fids)}–f{max(fids)}) ─')

    # Overall
    v_opt, v_in = e_opt[mask], e_in[mask]
    print(f'  2D alignment:  in mean={v_in.mean():.1f}  p50={np.median(v_in):.1f}  '
          f'→ opt mean={v_opt.mean():.1f}  p50={np.median(v_opt):.1f} px')

    # Outlier frames (estimator error > 200 px)
    bad = np.where(in_per_frame > 200)[0]
    if len(bad) > 0:
        print(f'  Frames with estimator error >200 px: {len(bad)}/{len(fids)}')
        for f in bad[:5]:
            print(f'    f{f:3d}: in={in_per_frame[f]:.0f} → opt={opt_per_frame[f]:.1f} px')
        if len(bad) > 5:
            print(f'    ... and {len(bad)-5} more')

    # Worst joints for input
    joint_errs_in = np.array([e_in[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else 0 for j in range(21)])
    worst = np.argsort(-joint_errs_in)[:5]
    print(f'  Worst joints (estimator): {", ".join(f"{JNAMES[j]}={joint_errs_in[j]:.0f}" for j in worst)} px')

    # Input cam_trans consistency (std across frames)
    t_std = np.std(in_trans[h][fids], axis=0)
    t_mean = np.mean(in_trans[h][fids], axis=0)
    print(f'  cam_trans:  mean=[{t_mean[0]:.3f}, {t_mean[1]:.3f}, {t_mean[2]:.3f}]  '
          f'std=[{t_std[0]:.3f}, {t_std[1]:.3f}, {t_std[2]:.3f}]')

    # Orientation jumps (detect sudden changes in per-frame predictions)
    o_diff = np.linalg.norm(np.diff(in_orients[h][fids], axis=0), axis=1)
    o_jump_thresh = np.percentile(o_diff, 95)
    jumps = np.where(o_diff > o_jump_thresh)[0]
    if len(jumps) > 0:
        print(f'  Orientation jumps (>p95={o_jump_thresh:.2f} rad): {len(jumps)} '
              f'at frames {[fids[j] for j in jumps[:5]]}{"..." if len(jumps) > 5 else ""}')

    # Per-joint improvement (opt vs input)
    joint_improve = joint_errs_in - np.array([e_opt[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else 0 for j in range(21)])
    best_improve = np.argsort(-joint_improve)[:3]
    print(f'  Most improved joints: {", ".join(f"{JNAMES[j]}={joint_improve[j]:.0f}" for j in best_improve)} px')


# ── Cross-hand ───────────────────────────────────────────────────
if B == 2:
    hh_dist = np.linalg.norm(opt['trans'][0] - opt['trans'][1], axis=1)
    print(f'\n─ Hand-hand distance: mean={hh_dist.mean():.2f}  min={hh_dist.min():.2f}  max={hh_dist.max():.2f} m')

    # Missing frames per hand
    fids0 = set(in_frame_ids[0])
    fids1 = set(in_frame_ids[1])
    missing_0 = sorted(fids1 - fids0)
    missing_1 = sorted(fids0 - fids1)
    if missing_0:
        print(f'  Left missing frames ({len(missing_0)}): f{min(missing_0)}–f{max(missing_0)}')
    if missing_1:
        print(f'  Right missing frames ({len(missing_1)}): f{min(missing_1)}–f{max(missing_1)}')


# ── Camera ───────────────────────────────────────────────────────
cam_pos = -np.einsum('tij,tj->ti', cam_R.transpose(0, 2, 1), cam_t)
cam_vel = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
cam_acc = np.linalg.norm(np.diff(cam_pos, n=2, axis=0), axis=1)
dr = np.linalg.norm(opt['delta_cam_R'], axis=1)
ws = opt['world_scale'][0, 0]

print(f'\n─ Camera:  world_scale={ws:.4f}  '
      f'vel mean={cam_vel.mean():.3f} max={cam_vel.max():.3f} m/f  '
      f'acc mean={cam_acc.mean():.3f} m/f²')
print(f'  Rotation adjustment: mean={dr.mean():.4f}  max={dr.max():.4f} rad  '
      f'p95={np.percentile(dr, 95):.4f} rad')


# ── Betas change ─────────────────────────────────────────────────
for h in range(B):
    b_d = np.linalg.norm(opt['betas'][h] - in_betas_arr[h])
    b_in = in_betas_arr[h]
    b_out = opt['betas'][h]
    top_dims = np.argsort(-np.abs(b_out - b_in))[:3]
    print(f'\n─ {HNAMES[h]} betas Δ={b_d:.3f}  '
          f'top changed dims: {", ".join(f"d{d}=({b_in[d]:+.2f}→{b_out[d]:+.2f})" for d in top_dims)}')


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print('\nmatplotlib not available.')
    sys.exit(0)

plt.rcParams.update({'font.size': 8, 'figure.dpi': 130})
C = ['#2196F3', '#FF5722']

# ── Figure 1: Per-frame error heatmap + timeline ──
fig, axes = plt.subplots(B, 2, figsize=(18, 3.5 * B), squeeze=False)

for h in range(B):
    e_in = np.sqrt(np.sum((in_joints2d_proj[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    e_opt = np.sqrt(np.sum((opt_joints2d[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    mask = in_joints2d[h][:, :, 2] > 0.3
    e_in[~mask] = np.nan
    e_opt[~mask] = np.nan

    # Heatmap: per-joint per-frame error (input)
    ax = axes[h, 0]
    im = ax.imshow(e_in.T, aspect='auto', cmap='hot', vmin=0, vmax=200, origin='lower')
    ax.set_xlabel('Frame'); ax.set_ylabel('Joint')
    ax.set_yticks(range(0, 21, 2)); ax.set_yticklabels([JNAMES[j] for j in range(0, 21, 2)])
    ax.set_title(f'{HNAMES[h]} hand — estimator 2D alignment (per-frame per-joint px error)')
    plt.colorbar(im, ax=ax, label='px')

    # Timeline: per-frame mean error
    ax = axes[h, 1]
    in_fm = np.array([e_in[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    opt_fm = np.array([e_opt[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    ax.fill_between(range(T), 0, in_fm, alpha=0.3, color='red', label='estimator')
    ax.plot(in_fm, color='red', lw=0.6, alpha=0.7)
    ax.plot(opt_fm, color='blue', lw=0.8, label='optimized')
    ax.axhline(y=10, color='gray', ls='--', alpha=0.4, lw=0.5)
    ax.set_xlabel('Frame'); ax.set_ylabel('Mean px error')
    ax.set_title(f'{HNAMES[h]} hand — per-frame mean 2D alignment')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    # Highlight outlier region
    bad = np.where(in_fm > 200)[0]
    if len(bad) > 0:
        ax.axvspan(bad[0], bad[-1], alpha=0.08, color='red')

plt.tight_layout()
p1 = os.path.join(args.out_dir, 'diagnose_frames.png')
plt.savefig(p1, bbox_inches='tight'); plt.close()
print(f'\nSaved {p1}')


# ── Figure 2: Hand trajectories + orientation ──
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Top-down (optimized, world space)
ax = fig.add_subplot(gs[0, 0])
for h in range(B):
    ax.plot(opt['trans'][h, :, 0], opt['trans'][h, :, 2], c=C[h], lw=0.6, label=f'{HNAMES[h]} opt')
    ax.scatter(*opt['trans'][h, 0, [0, 2]], c=C[h], s=30, marker='o')
    ax.scatter(*opt['trans'][h, -1, [0, 2]], c=C[h], s=30, marker='s')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Optimized translation (world, top-down)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

# Top-down (input, camera space — separate since it's a different frame)
ax = fig.add_subplot(gs[0, 1])
for h in range(B):
    fids = in_frame_ids[h]
    t = in_trans[h][fids]
    ax.plot(t[:, 0], t[:, 2], c=C[h], lw=0.6, label=f'{HNAMES[h]} in (cam space)')
    ax.scatter(*t[0, [0, 2]], c=C[h], s=30, marker='o')
    ax.scatter(*t[-1, [0, 2]], c=C[h], s=30, marker='s')
ax.set_xlabel('tx (m)'); ax.set_ylabel('tz (m)')
ax.set_title('Input cam_trans (camera space, top-down)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

# Height over time
ax = fig.add_subplot(gs[0, 2])
for h in range(B):
    t = opt['trans'][h]
    ax.plot(t[:, 1], c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    fids = in_frame_ids[h]
    ax.plot(fids, in_trans[h][fids, 1], c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
if B == 2:
    diff_y = opt['trans'][0, :, 1] - opt['trans'][1, :, 1]
    ax.plot(diff_y, c='#9C27B0', lw=0.5, alpha=0.5, label='height diff')
ax.set_xlabel('Frame'); ax.set_ylabel('Y (m)')
ax.set_title('Hand height (note: input in camera space)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Orientation over time
ax = fig.add_subplot(gs[1, 0])
for h in range(B):
    o_opt = np.linalg.norm(opt['root_orient'][h], axis=1)
    ax.plot(np.rad2deg(o_opt), c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    fids = in_frame_ids[h]
    o_in = np.linalg.norm(in_orients[h][fids], axis=1)
    ax.plot(fids, np.rad2deg(o_in), c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
ax.set_xlabel('Frame'); ax.set_ylabel('deg')
ax.set_title('Wrist orientation magnitude')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Orientation jumps (frame-to-frame delta)
ax = fig.add_subplot(gs[1, 1])
for h in range(B):
    o_diff_opt = np.linalg.norm(np.diff(opt['root_orient'][h], axis=0), axis=1)
    ax.plot(np.rad2deg(o_diff_opt), c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    fids = in_frame_ids[h]
    o_diff_in = np.linalg.norm(np.diff(in_orients[h][fids], axis=0), axis=1)
    ax.plot(fids[:-1], np.rad2deg(o_diff_in), c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
ax.set_xlabel('Frame'); ax.set_ylabel('deg/frame')
ax.set_title('Orientation frame-to-frame change (lower = smoother)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Camera + scale
ax = fig.add_subplot(gs[1, 2])
ax.plot(cam_pos[:, 0], cam_pos[:, 2], c='#4CAF50', lw=0.8, label='camera path')
ax.scatter(*cam_pos[0, [0, 2]], c='#4CAF50', s=40, marker='o', label='start')
ax.scatter(*cam_pos[-1, [0, 2]], c='#4CAF50', s=40, marker='s', label='end')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title(f'Camera trajectory (world_scale={ws:.3f})')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

plt.tight_layout()
p2 = os.path.join(args.out_dir, 'diagnose_trajectory.png')
plt.savefig(p2, bbox_inches='tight'); plt.close()
print(f'Saved {p2}')


# ── Figure 3: Per-joint breakdown ──
fig, axes = plt.subplots(B, 2, figsize=(20, 3.5 * B), squeeze=False)

for h in range(B):
    e_in = np.sqrt(np.sum((in_joints2d_proj[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    e_opt = np.sqrt(np.sum((opt_joints2d[h] - in_joints2d[h][:, :, :2]) ** 2, axis=-1))
    mask = in_joints2d[h][:, :, 2] > 0.3

    ax = axes[h, 0]
    joint_data_in = [e_in[:, j][mask[:, j]] for j in range(21)]
    bp = ax.boxplot(joint_data_in, patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1})
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral'); patch.set_alpha(0.6)
    ax.set_xticklabels(JNAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('px'); ax.set_title(f'{HNAMES[h]} — estimator')
    ax.grid(True, alpha=0.3, axis='y')
    ymax = max(np.percentile(d, 95) if len(d) > 0 else 0 for d in joint_data_in)
    ax.set_ylim(0, max(ymax * 1.1, 5))

    ax = axes[h, 1]
    joint_data_opt = [e_opt[:, j][mask[:, j]] for j in range(21)]
    bp = ax.boxplot(joint_data_opt, patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1})
    for patch in bp['boxes']:
        patch.set_facecolor(C[h]); patch.set_alpha(0.5)
    ax.set_xticklabels(JNAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('px'); ax.set_title(f'{HNAMES[h]} — optimized')
    ax.grid(True, alpha=0.3, axis='y')
    ymax = max(np.percentile(d, 95) if len(d) > 0 else 0 for d in joint_data_opt)
    ax.set_ylim(0, max(ymax * 1.5, 2))

plt.tight_layout()
p3 = os.path.join(args.out_dir, 'diagnose_joints.png')
plt.savefig(p3, bbox_inches='tight'); plt.close()
print(f'Saved {p3}')
