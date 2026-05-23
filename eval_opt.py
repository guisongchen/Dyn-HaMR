"""Evaluate optimized results with meaningful metrics and visualizations.

No ground truth exists — metrics measure:
  - 2D alignment: how well optimized 3D→2D matches the estimator's keypoints
  - 3D quality: trajectory smoothness, hand-hand distance, bone-length consistency
  - Camera quality: trajectory smoothness, rotation stability
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

opt = np.load(args.opt_npz)
with open(args.hands_json) as f:
    hands = json.load(f)['hands']
with open(args.cameras_json) as f:
    cams_data = json.load(f)

device = torch.device('cpu')
hand_model_r = MANO(batch_size=1, pose2rot=True, model_path='mano', is_rhand=True).to(device)

B, T = opt['pose_body'].shape[:2]
fx, fy, cx, cy = opt['intrins']
cam_R = opt['cam_R'][0]
cam_t = opt['cam_t'][0]


# ── Load input data ──────────────────────────────────────────────
in_joints2d, in_poses, in_orients, in_trans, in_betas = [], [], [], [], []
for h in hands:
    frames = h['frames']
    j2d = np.zeros((T, 21, 3), dtype=np.float32)
    pose = np.zeros((T, 15, 3), dtype=np.float32)
    orient = np.zeros((T, 3), dtype=np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    for f in frames:
        fid = f['frame_id']
        if fid < T:
            j2d[fid] = np.array(f['keypoints']['pose_keypoints_2d']).reshape(21, 3)
            pose[fid] = np.array(f['mano']['body_pose']).reshape(15, 3)
            orient[fid] = np.array(f['mano']['global_orient'])
            trans[fid] = np.array(f['mano']['cam_trans'])
    in_joints2d.append(j2d)
    in_poses.append(pose)
    in_orients.append(orient)
    in_trans.append(trans)
    in_betas.append(np.array(frames[0]['mano']['betas']))


# ── MANO forward pass ────────────────────────────────────────────
def run_mano_all(betas, poses, orients, trans, is_right_val):
    """Batch MANO forward: betas [T,10], poses [T,15,3], orients [T,3], trans [T,3]."""
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
    return j3d  # [T, 21, 3]


opt_joints3d = []
in_joints3d = []
for h in range(B):
    ir = int(opt['is_right'][h, 0])
    opt_joints3d.append(run_mano_all(
        np.tile(opt['betas'][h], (T, 1)), opt['pose_body'][h],
        opt['root_orient'][h], opt['trans'][h], ir))
    in_joints3d.append(run_mano_all(
        np.tile(in_betas[h], (T, 1)), in_poses[h],
        in_orients[h], in_trans[h], ir))


# ── 2D projection ────────────────────────────────────────────────
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
# METRICS
# ══════════════════════════════════════════════════════════════════
print('=' * 60)
print('EVALUATION  (no ground truth — estimator predictions only)')
print('=' * 60)

def alignment_err(gt_2d, pred_2d, conf_thresh=0.3):
    """Pixel distance between estimator 2D keypoints and projected 3D."""
    mask = gt_2d[:, :, 2] > conf_thresh
    diff = pred_2d - gt_2d[:, :, :2]
    return np.sqrt(np.sum(diff ** 2, axis=-1)), mask

for h in range(B):
    label = "right" if hands[h]["is_right"] else "left"
    err_opt, mask_opt = alignment_err(in_joints2d[h], opt_joints2d[h])
    err_in, mask_in = alignment_err(in_joints2d[h], in_joints2d_proj[h])
    v_opt, v_in = err_opt[mask_opt], err_in[mask_in]

    # 2D alignment (not "error" — this is the optimization loss)
    print(f'\nHand {h} ({label})  — {mask_opt.sum()} visible kp')
    print(f'  2D alignment (estimator, unscaled):  mean={v_in.mean():6.1f} px  p50={np.median(v_in):6.1f} px')
    print(f'  2D alignment (optimized):            mean={v_opt.mean():6.1f} px  p50={np.median(v_opt):6.1f} px')

    # 3D quality (no input comparison — different coordinate frames)
    t_vel = np.linalg.norm(np.diff(opt['trans'][h], axis=0), axis=1)
    t_acc = np.linalg.norm(np.diff(opt['trans'][h], n=2, axis=0), axis=1)
    print(f'  Trans velocity:  mean={t_vel.mean():.4f}  max={t_vel.max():.4f} m/f')
    print(f'  Trans accel:     mean={t_acc.mean():.4f}  max={t_acc.max():.4f} m/f²')

    p_vel = np.linalg.norm(np.diff(opt['pose_body'][h], axis=0), axis=(1, 2))
    p_acc = np.linalg.norm(np.diff(opt['pose_body'][h], n=2, axis=0), axis=(1, 2))
    print(f'  Pose velocity:   mean={p_vel.mean():.4f}  max={p_vel.max():.4f} rad/f')
    print(f'  Pose accel:      mean={p_acc.mean():.4f}  max={p_acc.max():.4f} rad/f²')

    # Bone-length consistency: parent-child distance std across frames
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
    def bone_lengths(j3d):
        bl = []
        for c, p in enumerate(parents):
            if p >= 0:
                bl.append(np.linalg.norm(j3d[:, c] - j3d[:, p], axis=1))
        return np.array(bl)  # [n_bones, T]
    bl_opt = bone_lengths(opt_joints3d[h])
    bl_in = bone_lengths(in_joints3d[h])
    bl_std_opt = np.std(bl_opt, axis=1).mean()
    bl_std_in = np.std(bl_in, axis=1).mean()
    print(f'  Bone-length std  (estimator → opt): {bl_std_in:.4f} → {bl_std_opt:.4f} m')

    # Betas change
    b_d = np.linalg.norm(opt['betas'][h] - in_betas[h])
    print(f'  Betas Δ from input: {b_d:.4f}')

# Hand-hand distance
if B == 2:
    hh_dist = np.linalg.norm(opt['trans'][0] - opt['trans'][1], axis=1)
    print(f'\nHand-hand distance:  mean={hh_dist.mean():.3f}  min={hh_dist.min():.3f}  max={hh_dist.max():.3f} m')

# Camera trajectory smoothness
cam_pos = -np.einsum('tij,tj->ti', cam_R.transpose(0, 2, 1), cam_t)
cam_vel = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
cam_acc = np.linalg.norm(np.diff(cam_pos, n=2, axis=0), axis=1)
print(f'Camera velocity:  mean={cam_vel.mean():.4f}  max={cam_vel.max():.4f} m/f')
print(f'Camera accel:     mean={cam_acc.mean():.4f}  max={cam_acc.max():.4f} m/f²')

ws = opt['world_scale'][0, 0]
dr = np.linalg.norm(opt['delta_cam_R'], axis=1)
print(f'world_scale: {ws:.4f}  |  cam_R adjustment: mean={dr.mean():.4f} rad/f')


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('\nmatplotlib not available, skipping plots.')
    sys.exit(0)

plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})
C = ['#2196F3', '#FF5722']
H = ['Left hand', 'Right hand']

fig, axes = plt.subplots(3, 2, figsize=(16, 13))

# 1 — 2D alignment over time
ax = axes[0, 0]
for h in range(B):
    err, mask = alignment_err(in_joints2d[h], opt_joints2d[h])
    fm = np.array([err[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    ax.plot(fm, c=C[h], lw=0.6, label=H[h])
ax.set_xlabel('Frame'); ax.set_ylabel('px')
ax.set_title('2D keypoint alignment (optimized 3D → 2D projection vs estimator kp)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2 — Per-joint alignment
ax = axes[0, 1]
jnames = ['Wr','I1','I2','I3','M1','M2','M3','P1','P2','P3','R1','R2','R3','T1','T2','T3','J17','J18','J19','J20','J21']
for h in range(B):
    err, mask = alignment_err(in_joints2d[h], opt_joints2d[h])
    pj = [err[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else 0 for j in range(21)]
    ax.bar(np.arange(21) + h * 0.3, pj, width=0.3, color=C[h], alpha=0.8, label=H[h])
ax.set_xticks(np.arange(21) + 0.15); ax.set_xticklabels(jnames, rotation=45, ha='right', fontsize=6)
ax.set_ylabel('px'); ax.set_title('Per-joint 2D alignment'); ax.legend(fontsize=8)

# 3 — Hand translation (XZ world-space top-down view)
ax = axes[1, 0]
for h in range(B):
    ax.plot(opt['trans'][h, :, 0], opt['trans'][h, :, 2], c=C[h], lw=0.8, label=H[h])
    ax.scatter(*opt['trans'][h, 0, [0, 2]], c=C[h], s=40, marker='o')
    ax.scatter(*opt['trans'][h, -1, [0, 2]], c=C[h], s=40, marker='s')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Hand translation — top-down view (world space)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axis('equal')

# 4 — Translation velocity + hand-hand distance
ax = axes[1, 1]
for h in range(B):
    v = np.linalg.norm(np.diff(opt['trans'][h], axis=0), axis=1)
    ax.plot(v, c=C[h], lw=0.6, alpha=0.7, label=f'{H[h]} speed')
if B == 2:
    hh = np.linalg.norm(opt['trans'][0] - opt['trans'][1], axis=1)
    ax.plot(hh, c='#9C27B0', lw=1.0, label='hand-hand distance')
ax.set_xlabel('Frame'); ax.set_ylabel('m or m/f')
ax.set_title('Translation speed & hand-hand distance')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5 — Pose velocity (per-frame angular change)
ax = axes[2, 0]
for h in range(B):
    v = np.linalg.norm(np.diff(opt['pose_body'][h], axis=0), axis=(1, 2))
    ax.plot(v, c=C[h], lw=0.4, alpha=0.7, label=H[h])
ax.set_xlabel('Frame'); ax.set_ylabel('rad/frame')
ax.set_title('Pose velocity (per-frame angular change)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 6 — Camera trajectory
ax = axes[2, 1]
ax.plot(cam_pos[:, 0], cam_pos[:, 2], c='#4CAF50', lw=0.8)
ax.scatter(*cam_pos[0, [0, 2]], c='#4CAF50', s=40, marker='o', label='start')
ax.scatter(*cam_pos[-1, [0, 2]], c='#4CAF50', s=40, marker='s', label='end')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Camera trajectory (world space)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axis('equal')

plt.tight_layout()
p1 = os.path.join(args.out_dir, 'eval_metrics.png')
plt.savefig(p1, bbox_inches='tight'); plt.close()
print(f'\nSaved {p1}')

# ── Second figure ──
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# 2D alignment improvement over input
ax = axes2[0, 0]
for h in range(B):
    e_opt, m = alignment_err(in_joints2d[h], opt_joints2d[h])
    e_in, _ = alignment_err(in_joints2d[h], in_joints2d_proj[h])
    opt_fm = [e_opt[t, m[t]].mean() if m[t].sum() > 0 else np.nan for t in range(T)]
    in_fm = [e_in[t, m[t]].mean() if m[t].sum() > 0 else np.nan for t in range(T)]
    ax.plot(np.array(opt_fm) - np.array(in_fm), c=C[h], lw=0.6, label=H[h])
ax.axhline(0, c='black', ls='--', alpha=0.3)
ax.set_xlabel('Frame'); ax.set_ylabel('Δ px (optimized − estimator projection)')
ax.set_title('2D alignment improvement over estimator\'s own 3D params')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Hand height
ax = axes2[0, 1]
for h in range(B):
    ax.plot(opt['trans'][h, :, 1], c=C[h], lw=0.8, alpha=0.7, label=f'{H[h]} Y')
if B == 2:
    ax.plot(opt['trans'][0, :, 1] - opt['trans'][1, :, 1], c='#9C27B0', lw=0.6, alpha=0.5, label='height diff')
ax.set_xlabel('Frame'); ax.set_ylabel('Y (m)')
ax.set_title('Hand height (world-space Y)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Pose velocity + acceleration
ax = axes2[1, 0]
for h in range(B):
    v = np.linalg.norm(np.diff(opt['pose_body'][h], axis=0), axis=(1, 2))
    ax.plot(v, c=C[h], lw=0.6, alpha=0.7, label=f'{H[h]} vel')
ax.set_xlabel('Frame'); ax.set_ylabel('rad/frame')
ax.set_title('Pose velocity (per-frame angular change)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Camera velocity & rotation adjustment
ax = axes2[1, 1]
ax2 = ax.twinx()
ax.plot(cam_vel, c='#4CAF50', lw=0.8, label='camera velocity (m/f)')
ax2.plot(dr, c='#FF9800', lw=0.6, alpha=0.7, label='cam_R adjustment (rad)')
ax.set_xlabel('Frame'); ax.set_ylabel('m/frame'); ax2.set_ylabel('rad')
ax.set_title('Camera velocity + rotation refinement')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
p2 = os.path.join(args.out_dir, 'eval_compare.png')
plt.savefig(p2, bbox_inches='tight'); plt.close()
print(f'Saved {p2}')
