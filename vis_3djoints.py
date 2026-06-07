#!/usr/bin/env python3
"""3D visualization of optimized hand joints from hands.json.

Usage:
    # Interactive window
    python vis_3djoints.py --hands outputs/hands.json

    # Save single-frame image
    python vis_3djoints.py --hands outputs/hands.json --frame 50 --out viz3d.png

    # Save frame-by-frame animation (GIF or MP4)
    python vis_3djoints.py --hands outputs/hands.json --out viz3d.gif
    python vis_3djoints.py --hands outputs/hands.json --out viz3d.mp4 --fps 15

    # Show specific hands only (0-indexed)
    python vis_3djoints.py --hands outputs/hands.json --hand_idx 0 1
"""
import os
import sys
import json
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Joint layout (21 OpenPose hand joints) ─────────────────────────
JNAMES = [
    'Wrist',
    'T1', 'T2', 'T3', 'Ttip',
    'I1', 'I2', 'I3', 'Itip',
    'M1', 'M2', 'M3', 'Mtip',
    'R1', 'R2', 'R3', 'Rtip',
    'P1', 'P2', 'P3', 'Ptip',
]

HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

FINGER_RANGES = {
    'Thumb':  (1, 5),
    'Index':  (5, 9),
    'Middle': (9, 13),
    'Ring':   (13, 17),
    'Pinky':  (17, 21),
}

FINGER_COLORS = {
    'Thumb':  '#e6194b',
    'Index':  '#3cb44b',
    'Middle': '#ffe119',
    'Ring':   '#4363d8',
    'Pinky':  '#f58231',
}
WRIST_COLOR = '#800000'

# Axis limits fallback range (metres)
DEFAULT_LIM = 0.25


def load_hands(path):
    with open(path) as f:
        data = json.load(f)
    try:
        hands = data['hands']
    except KeyError:
        raise ValueError(f"{path} does not contain 'hands' key; "
                         f"expected canonical format from export_canonical_json")
    return hands


def extract_joints3d(hands, hand_indices=None):
    """Return list of (is_right, joints[T,21,3]) for selected hands."""
    results = []
    idxs = (hand_indices if hand_indices is not None
            else list(range(len(hands))))
    for i in idxs:
        h = hands[i]
        ir = h['is_right']
        frames = sorted(h['frames'], key=lambda f: f['frame_id'])
        T = max(f['frame_id'] for f in frames) + 1
        j3d = np.zeros((T, 21, 3))
        for f in frames:
            fid = f['frame_id']
            k = f.get('keypoints', {})
            raw = k.get('pose_keypoints_3d', [])
            if len(raw) == 63:
                j3d[fid] = np.array(raw).reshape(21, 3)
            else:
                j3d[fid] = np.full((21, 3), np.nan)
        results.append((ir, j3d))
    return results


def set_axes_equal(ax):
    """Make axes of a 3D plot have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_single_frame(joints_list, ax, frame_idx=0, elev=20, azim=-60,
                      title=''):
    """Draw joints + skeleton for one frame onto a 3D Axes."""
    for is_right, j3d in joints_list:
        if frame_idx >= len(j3d):
            continue
        pts = j3d[frame_idx]  # [21, 3]
        if np.all(np.isnan(pts)):
            continue

        side = 'R' if is_right else 'L'

        # wrist
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                   c=WRIST_COLOR, s=80, marker='o', zorder=3)
        ax.text(pts[0, 0] + 0.015, pts[0, 1] + 0.015, pts[0, 2],
                side, fontsize=8, fontweight='bold', color=WRIST_COLOR)

        # fingers
        for finger, (s, e) in FINGER_RANGES.items():
            color = FINGER_COLORS[finger]
            ax.scatter(pts[s:e, 0], pts[s:e, 1], pts[s:e, 2],
                       c=color, s=30, marker='o', zorder=2)

        # skeleton lines
        for p, c in HAND_SKELETON:
            ax.plot([pts[p, 0], pts[c, 0]],
                    [pts[p, 1], pts[c, 1]],
                    [pts[p, 2], pts[c, 2]],
                    color='gray', linewidth=1, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    set_axes_equal(ax)


def run_interactive(joints_list, out_path=None):
    """Pop up an interactive matplotlib window."""
    T = max(j3d.shape[0] for _, j3d in joints_list)
    print(f"Total frames: {T}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    frame_idx = [0]

    def draw():
        ax.clear()
        plot_single_frame(joints_list, ax, frame_idx=frame_idx[0],
                          title=f'Frame {frame_idx[0]}/{T - 1}')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            frame_idx[0] = min(frame_idx[0] + 1, T - 1)
        elif event.key == 'left':
            frame_idx[0] = max(frame_idx[0] - 1, 0)
        elif event.key == 'up':
            frame_idx[0] = min(frame_idx[0] + 10, T - 1)
        elif event.key == 'down':
            frame_idx[0] = max(frame_idx[0] - 10, 0)
        draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()


def run_animation(joints_list, out_path, fps=10):
    """Save a GIF/MP4 animation showing all frames."""
    T = max(j3d.shape[0] for _, j3d in joints_list)
    print(f"Generating animation ({T} frames) -> {out_path}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(t):
        ax.clear()
        plot_single_frame(joints_list, ax, frame_idx=t,
                          title=f'Frame {t}/{T - 1}')

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps,
                        repeat=False)
    ext = os.path.splitext(out_path)[1].lower()
    writer = 'ffmpeg' if ext in ('.mp4', '.mov') else 'pillow'
    ani.save(out_path, writer=writer, fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_static(joints_list, out_path, frame_idx=0):
    """Save a single frame as PNG."""
    print(f"Rendering frame {frame_idx} -> {out_path}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_single_frame(joints_list, ax, frame_idx=frame_idx)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='3D visualization of hand joints from hands.json')
    parser.add_argument('--hands', required=True,
                        help='Path to optimized hands.json')
    parser.add_argument('--out', default=None,
                        help='Output file (.png for single frame, '
                             '.gif/.mp4 for animation). If omitted, '
                             'opens an interactive window.')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index for single-frame output (default 0)')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for animation output (default 10)')
    parser.add_argument('--hand_idx', type=int, nargs='*', default=None,
                        help='Hand indices to show (0-based). If omitted, '
                             'all hands are shown.')
    args = parser.parse_args()

    hands = load_hands(args.hands)
    joints_list = extract_joints3d(hands, args.hand_idx)
    if not joints_list:
        print("No hands with valid 3D joints found.", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        run_interactive(joints_list)
    else:
        ext = os.path.splitext(args.out)[1].lower()
        if ext in ('.gif', '.mp4', '.mov', '.webm'):
            run_animation(joints_list, args.out, fps=args.fps)
        else:
            run_static(joints_list, args.out, frame_idx=args.frame)


if __name__ == '__main__':
    main()
