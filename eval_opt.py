"""Diagnose optimization quality — compare input vs optimized canonical JSON.

Reads from the portable cameras.json + hands.json format (same as input).
No .npz dependency.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO

parser = argparse.ArgumentParser()
parser.add_argument('--input_cameras', default='demo_data/cameras.json')
parser.add_argument('--input_hands',   default='demo_data/hands.json')
parser.add_argument('--opt_cameras',   default='outputs/cameras.json')
parser.add_argument('--opt_hands',     default='outputs/hands.json')
parser.add_argument('--out_dir',       default='outputs/eval')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device('cpu')
hand_model_r = MANO(batch_size=1, pose2rot=True, model_path='mano', is_rhand=True).to(device)


# ── Helpers ──────────────────────────────────────────────────────
def load_cameras(path):
    with open(path) as f:
        c = json.load(f)
    return {
        'w2c':      np.array(c['w2c']),          # [T, 4, 4]
        'intrins':  np.array(c['intrins']),       # [4]
        'height':   c['height'],
        'width':    c['width'],
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
            m = f['mano']; k = f['keypoints']
            j2d[fid] = np.array(k['pose_keypoints_2d']).reshape(21, 3)
            pose[fid] = np.array(m['body_pose']).reshape(15, 3)
            orient[fid] = np.array(m['global_orient'])
            trans[fid] = np.array(m['cam_trans'])
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
                global_orient=torch.tensor(orients[i:i+1], dtype=torch.float32),
                hand_pose=torch.tensor(poses[i:i+1].reshape(1, 45), dtype=torch.float32),
                transl=torch.tensor(trans[i:i+1], dtype=torch.float32),
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
        j_cam = np.einsum('tij,tkj->tki', R, j3d) + t[:, None, :]  # [T, 21, 3]
    else:
        j_cam = j3d
    x = fx * j_cam[:, :, 0] / j_cam[:, :, 2] + cx
    y = fy * j_cam[:, :, 1] / j_cam[:, :, 2] + cy
    return np.stack([x, y], axis=-1)


# ── Load data ────────────────────────────────────────────────────
in_cams = load_cameras(args.input_cameras)
in_hands = load_hands(args.input_hands)
opt_cams = load_cameras(args.opt_cameras)
opt_hands = load_hands(args.opt_hands)

B = len(opt_hands)
T = opt_cams['num_frames']
fx, fy, cx, cy = opt_cams['intrins'].astype(np.float64)

JNAMES = ['Wrist','I1','I2','I3','M1','M2','M3','P1','P2','P3',
          'R1','R2','R3','T1','T2','T3','Tp1','Tp2','Tp3','Tp4','Tp5']
HNAMES = ['Left', 'Right']
C = ['#2196F3', '#FF5722']

# Compute 3D joints for both input and optimized
in_joints3d, opt_joints3d = [], []
for h in range(B):
    in_joints3d.append(run_mano(in_hands[h]['betas'], in_hands[h]['pose'],
                                 in_hands[h]['orient'], in_hands[h]['trans'],
                                 in_hands[h]['is_right']))
    opt_joints3d.append(run_mano(opt_hands[h]['betas'], opt_hands[h]['pose'],
                                  opt_hands[h]['orient'], opt_hands[h]['trans'],
                                  opt_hands[h]['is_right']))

# Project to 2D
# Input: cam_trans is camera-space → project without W2C
# Optimized: trans is world-space → project with W2C
opt_joints2d = [project(j, opt_cams['w2c'], opt_cams['intrins']) for j in opt_joints3d]
in_joints2d_proj = [project(j, None, in_cams['intrins']) for j in in_joints3d]


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
print('=' * 70)
print('DIAGNOSTIC REPORT')
print('=' * 70)

for h in range(B):
    ih = in_hands[h]; oh = opt_hands[h]
    label = HNAMES[h]

    e_opt = np.sqrt(np.sum((opt_joints2d[h] - ih['joints2d'][:, :, :2]) ** 2, axis=-1))
    e_in  = np.sqrt(np.sum((in_joints2d_proj[h] - ih['joints2d'][:, :, :2]) ** 2, axis=-1))
    mask = ih['joints2d'][:, :, 2] > 0.3

    fids = sorted([f['frame_id'] for f in
                   json.load(open(args.input_hands))['hands'][h]['frames']])
    opt_per_frame = np.array([e_opt[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    in_per_frame  = np.array([e_in[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])

    v_opt, v_in = e_opt[mask], e_in[mask]
    print(f'\n─ Hand: {label} ({len(fids)} input frames, f{min(fids)}–f{max(fids)}) ─')
    print(f'  2D alignment:  in mean={v_in.mean():.1f}  p50={np.median(v_in):.1f}  '
          f'→ opt mean={v_opt.mean():.1f}  p50={np.median(v_opt):.1f} px')

    bad = np.where(in_per_frame > 200)[0]
    if len(bad) > 0:
        print(f'  Frames with estimator error >200 px: {len(bad)}/{len(fids)}')
        for f in bad[:5]:
            print(f'    f{f:3d}: in={in_per_frame[f]:.0f} → opt={opt_per_frame[f]:.1f} px')
        if len(bad) > 5:
            print(f'    ... and {len(bad)-5} more')

    joint_errs_in = np.array([e_in[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else 0 for j in range(21)])
    worst = np.argsort(-joint_errs_in)[:5]
    print(f'  Worst joints (estimator): {", ".join(f"{JNAMES[j]}={joint_errs_in[j]:.0f}" for j in worst)} px')

    t_data = ih['trans'][fids]
    t_std = np.std(t_data, axis=0)
    print(f'  cam_trans:  mean=[{t_data.mean(0)[0]:.3f}, {t_data.mean(0)[1]:.3f}, {t_data.mean(0)[2]:.3f}]  '
          f'std=[{t_std[0]:.3f}, {t_std[1]:.3f}, {t_std[2]:.3f}]')

    o_diff = np.linalg.norm(np.diff(ih['orient'][fids], axis=0), axis=1)
    o_jump_thresh = np.percentile(o_diff, 95)
    jumps = np.where(o_diff > o_jump_thresh)[0]
    if len(jumps) > 0:
        print(f'  Orientation jumps (>p95={o_jump_thresh:.2f} rad): {len(jumps)} '
              f'at frames {[fids[j] for j in jumps[:5]]}{"..." if len(jumps)>5 else ""}')

    joint_improve = joint_errs_in - np.array([e_opt[:, j][mask[:, j]].mean() if mask[:, j].sum() > 0 else 0 for j in range(21)])
    best_improve = np.argsort(-joint_improve)[:3]
    print(f'  Most improved joints: {", ".join(f"{JNAMES[j]}={joint_improve[j]:.0f}" for j in best_improve)} px')


# ── Cross-hand ───────────────────────────────────────────────────
if B == 2:
    in_fids0 = set(f['frame_id'] for f in json.load(open(args.input_hands))['hands'][0]['frames'])
    in_fids1 = set(f['frame_id'] for f in json.load(open(args.input_hands))['hands'][1]['frames'])
    hh_dist = np.linalg.norm(opt_hands[0]['trans'] - opt_hands[1]['trans'], axis=1)
    print(f'\n─ Hand-hand distance: mean={hh_dist.mean():.2f}  min={hh_dist.min():.2f}  max={hh_dist.max():.2f} m')
    missing_0 = sorted(in_fids1 - in_fids0)
    missing_1 = sorted(in_fids0 - in_fids1)
    if missing_0:
        print(f'  Left missing frames ({len(missing_0)}): f{min(missing_0)}–f{max(missing_0)}')
    if missing_1:
        print(f'  Right missing frames ({len(missing_1)}): f{min(missing_1)}–f{max(missing_1)}')


# ── Camera ───────────────────────────────────────────────────────
w2c_in = in_cams['w2c']
w2c_out = opt_cams['w2c']
cam_pos_in  = -np.einsum('tij,tj->ti',  w2c_in[:, :3, :3].transpose(0, 2, 1),  w2c_in[:, :3, 3])
cam_pos_out = -np.einsum('tij,tj->ti', w2c_out[:, :3, :3].transpose(0, 2, 1), w2c_out[:, :3, 3])
cam_vel = np.linalg.norm(np.diff(cam_pos_out, axis=0), axis=1)
cam_acc = np.linalg.norm(np.diff(cam_pos_out, n=2, axis=0), axis=1)
dr = np.array([np.linalg.norm(w2c_out[t, :3, :3] - w2c_in[t, :3, :3]) for t in range(T)])

print(f'\n─ Camera:  vel mean={cam_vel.mean():.3f} max={cam_vel.max():.3f} m/f  '
      f'acc mean={cam_acc.mean():.3f} m/f²')
print(f'  Rotation change from input: mean={dr.mean():.4f}  max={dr.max():.4f} rad  '
      f'p95={np.percentile(dr, 95):.4f} rad')

# Betas
for h in range(B):
    b_in = in_hands[h]['betas']; b_out = opt_hands[h]['betas']
    b_d = np.linalg.norm(b_out - b_in)
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
except ImportError:
    print('\nmatplotlib not available.'); sys.exit(0)

plt.rcParams.update({'font.size': 8, 'figure.dpi': 130})


# ── Figure 1: Per-frame error heatmap + timeline ──
fig, axes = plt.subplots(B, 2, figsize=(18, 3.5 * B), squeeze=False)

for h in range(B):
    e_in  = np.sqrt(np.sum((in_joints2d_proj[h] - in_hands[h]['joints2d'][:, :, :2]) ** 2, axis=-1))
    e_opt = np.sqrt(np.sum((opt_joints2d[h] - in_hands[h]['joints2d'][:, :, :2]) ** 2, axis=-1))
    mask = in_hands[h]['joints2d'][:, :, 2] > 0.3
    e_in[~mask] = np.nan; e_opt[~mask] = np.nan

    ax = axes[h, 0]
    im = ax.imshow(e_in.T, aspect='auto', cmap='hot', vmin=0, vmax=200, origin='lower')
    ax.set_xlabel('Frame'); ax.set_ylabel('Joint')
    ax.set_yticks(range(0, 21, 2)); ax.set_yticklabels([JNAMES[j] for j in range(0, 21, 2)])
    ax.set_title(f'{HNAMES[h]} — estimator 2D alignment (px)')
    plt.colorbar(im, ax=ax, label='px')

    ax = axes[h, 1]
    in_fm  = np.array([e_in[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    opt_fm = np.array([e_opt[t, mask[t]].mean() if mask[t].sum() > 0 else np.nan for t in range(T)])
    ax.fill_between(range(T), 0, in_fm, alpha=0.3, color='red', label='estimator')
    ax.plot(in_fm, color='red', lw=0.6, alpha=0.7)
    ax.plot(opt_fm, color='blue', lw=0.8, label='optimized')
    ax.axhline(y=10, color='gray', ls='--', alpha=0.4, lw=0.5)
    ax.set_xlabel('Frame'); ax.set_ylabel('Mean px')
    ax.set_title(f'{HNAMES[h]} — per-frame mean 2D alignment')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    bad = np.where(in_fm > 200)[0]
    if len(bad) > 0:
        ax.axvspan(bad[0], bad[-1], alpha=0.08, color='red')

plt.tight_layout()
p1 = os.path.join(args.out_dir, 'diagnose_frames.png')
plt.savefig(p1, bbox_inches='tight'); plt.close()
print(f'\nSaved {p1}')


# ── Figure 2: Trajectory + orientation ──
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
for h in range(B):
    t = opt_hands[h]['trans']
    ax.plot(t[:, 0], t[:, 2], c=C[h], lw=0.6, label=f'{HNAMES[h]} opt')
    ax.scatter(*t[0, [0, 2]], c=C[h], s=30, marker='o')
    ax.scatter(*t[-1, [0, 2]], c=C[h], s=30, marker='s')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Optimized translation (world, top-down)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

ax = fig.add_subplot(gs[0, 1])
for h in range(B):
    fids = sorted([f['frame_id'] for f in json.load(open(args.input_hands))['hands'][h]['frames']])
    t = in_hands[h]['trans'][fids]
    ax.plot(t[:, 0], t[:, 2], c=C[h], lw=0.6, label=f'{HNAMES[h]} in (cam)')
    ax.scatter(*t[0, [0, 2]], c=C[h], s=30, marker='o')
    ax.scatter(*t[-1, [0, 2]], c=C[h], s=30, marker='s')
ax.set_xlabel('tx (m)'); ax.set_ylabel('tz (m)')
ax.set_title('Input cam_trans (camera space)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

ax = fig.add_subplot(gs[0, 2])
for h in range(B):
    fids = sorted([f['frame_id'] for f in json.load(open(args.input_hands))['hands'][h]['frames']])
    ax.plot(opt_hands[h]['trans'][:, 1], c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    ax.plot(fids, in_hands[h]['trans'][fids, 1], c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
ax.set_xlabel('Frame'); ax.set_ylabel('Y (m)')
ax.set_title('Hand height (note: input in camera space)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
for h in range(B):
    o_opt = np.linalg.norm(opt_hands[h]['orient'], axis=1)
    ax.plot(np.rad2deg(o_opt), c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    fids = sorted([f['frame_id'] for f in json.load(open(args.input_hands))['hands'][h]['frames']])
    o_in = np.linalg.norm(in_hands[h]['orient'][fids], axis=1)
    ax.plot(fids, np.rad2deg(o_in), c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
ax.set_xlabel('Frame'); ax.set_ylabel('deg')
ax.set_title('Wrist orientation magnitude')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
for h in range(B):
    o_diff_opt = np.linalg.norm(np.diff(opt_hands[h]['orient'], axis=0), axis=1)
    ax.plot(np.rad2deg(o_diff_opt), c=C[h], lw=0.8, label=f'{HNAMES[h]} opt')
    fids = sorted([f['frame_id'] for f in json.load(open(args.input_hands))['hands'][h]['frames']])
    o_diff_in = np.linalg.norm(np.diff(in_hands[h]['orient'][fids], axis=0), axis=1)
    ax.plot(fids[:-1], np.rad2deg(o_diff_in), c=C[h], lw=0.4, ls='--', alpha=0.5, label=f'{HNAMES[h]} in')
ax.set_xlabel('Frame'); ax.set_ylabel('deg/frame')
ax.set_title('Orientation frame-to-frame change')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(cam_pos_out[:, 0], cam_pos_out[:, 2], c='#4CAF50', lw=0.8, label='opt camera')
ax.plot(cam_pos_in[:, 0], cam_pos_in[:, 2], c='#4CAF50', lw=0.3, ls='--', alpha=0.4, label='in camera')
ax.scatter(*cam_pos_out[0, [0, 2]], c='#4CAF50', s=40, marker='o')
ax.scatter(*cam_pos_out[-1, [0, 2]], c='#4CAF50', s=40, marker='s')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Camera trajectory')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axis('equal')

plt.tight_layout()
p2 = os.path.join(args.out_dir, 'diagnose_trajectory.png')
plt.savefig(p2, bbox_inches='tight'); plt.close()
print(f'Saved {p2}')


# ── Figure 3: Per-joint breakdown ──
fig, axes = plt.subplots(B, 2, figsize=(20, 3.5 * B), squeeze=False)

for h in range(B):
    e_in  = np.sqrt(np.sum((in_joints2d_proj[h] - in_hands[h]['joints2d'][:, :, :2]) ** 2, axis=-1))
    e_opt = np.sqrt(np.sum((opt_joints2d[h] - in_hands[h]['joints2d'][:, :, :2]) ** 2, axis=-1))
    mask = in_hands[h]['joints2d'][:, :, 2] > 0.3

    ax = axes[h, 0]
    joint_data_in = [e_in[:, j][mask[:, j]] for j in range(21)]
    bp = ax.boxplot(joint_data_in, patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1})
    for patch in bp['boxes']: patch.set_facecolor('lightcoral'); patch.set_alpha(0.6)
    ax.set_xticklabels(JNAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('px'); ax.set_title(f'{HNAMES[h]} — estimator')
    ax.grid(True, alpha=0.3, axis='y')
    ymax = max(np.percentile(d, 95) if len(d) > 0 else 0 for d in joint_data_in)
    ax.set_ylim(0, max(ymax * 1.1, 5))

    ax = axes[h, 1]
    joint_data_opt = [e_opt[:, j][mask[:, j]] for j in range(21)]
    bp = ax.boxplot(joint_data_opt, patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1})
    for patch in bp['boxes']: patch.set_facecolor(C[h]); patch.set_alpha(0.5)
    ax.set_xticklabels(JNAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('px'); ax.set_title(f'{HNAMES[h]} — optimized')
    ax.grid(True, alpha=0.3, axis='y')
    ymax = max(np.percentile(d, 95) if len(d) > 0 else 0 for d in joint_data_opt)
    ax.set_ylim(0, max(ymax * 1.5, 2))

plt.tight_layout()
p3 = os.path.join(args.out_dir, 'diagnose_joints.png')
plt.savefig(p3, bbox_inches='tight'); plt.close()
print(f'Saved {p3}')
