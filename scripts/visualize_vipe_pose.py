#!/usr/bin/env python3
"""Visualize VIPE camera-pose trajectory from NPZ files."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_poses(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load poses and frame indices from a VIPE NPZ file."""
    data = np.load(npz_path)
    return data["data"], data["inds"]  # (N, 4, 4), (N,)


def camera_centers_from_poses(poses: np.ndarray) -> np.ndarray:
    """Compute camera centers assuming poses are world->camera [R|t]."""
    centers = []
    for p in poses:
        R = p[:3, :3]
        t = p[:3, 3]
        centers.append(-R.T @ t)
    return np.array(centers)


def visualize(poses: np.ndarray, _inds: np.ndarray, out_path: Path, step: int = 30) -> None:
    """Render a static multi-panel trajectory figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(poses)
    centers = camera_centers_from_poses(poses)
    frames = np.arange(n)

    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory with camera axes
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    sc = ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=frames, cmap="viridis", s=10)
    ax1.plot(centers[:, 0], centers[:, 1], centers[:, 2], "k-", alpha=0.3, linewidth=0.5)
    scale = 0.3
    for i in range(0, n, step):
        R = poses[i, :3, :3]
        c = centers[i]
        ax1.plot([c[0], c[0] + scale * R[0, 0]], [c[1], c[1] + scale * R[1, 0]], [c[2], c[2] + scale * R[2, 0]], "r-", linewidth=1)
        ax1.plot([c[0], c[0] + scale * R[0, 1]], [c[1], c[1] + scale * R[1, 1]], [c[2], c[2] + scale * R[2, 1]], "g-", linewidth=1)
        ax1.plot([c[0], c[0] + scale * R[0, 2]], [c[1], c[1] + scale * R[1, 2]], [c[2], c[2] + scale * R[2, 2]], "b-", linewidth=1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Camera Trajectory")
    plt.colorbar(sc, ax=ax1, label="Frame")

    # Top-down (XY)
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(centers[:, 0], centers[:, 1], c=frames, cmap="viridis", s=10)
    ax2.plot(centers[:, 0], centers[:, 1], "k-", alpha=0.3, linewidth=0.5)
    for i in range(0, n, step):
        R = poses[i, :3, :3]
        c = centers[i]
        ax2.arrow(c[0], c[1], scale * R[0, 2], scale * R[1, 2], head_width=0.1, color="blue", alpha=0.7)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-Down View (XY)")
    ax2.set_aspect("equal")
    plt.colorbar(sc2, ax=ax2, label="Frame")

    # Side (XZ)
    ax3 = fig.add_subplot(2, 3, 3)
    sc3 = ax3.scatter(centers[:, 0], centers[:, 2], c=frames, cmap="viridis", s=10)
    ax3.plot(centers[:, 0], centers[:, 2], "k-", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("Side View (XZ)")
    ax3.set_aspect("equal")
    plt.colorbar(sc3, ax=ax3, label="Frame")

    # Side (YZ)
    ax4 = fig.add_subplot(2, 3, 4)
    sc4 = ax4.scatter(centers[:, 1], centers[:, 2], c=frames, cmap="viridis", s=10)
    ax4.plot(centers[:, 1], centers[:, 2], "k-", alpha=0.3, linewidth=0.5)
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_title("Side View (YZ)")
    ax4.set_aspect("equal")
    plt.colorbar(sc4, ax=ax4, label="Frame")

    # Translation magnitude from origin
    ax5 = fig.add_subplot(2, 3, 5)
    trans = poses[:, :3, 3]
    mags = np.linalg.norm(trans, axis=1)
    ax5.plot(frames, mags, linewidth=1)
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("|t| (m)")
    ax5.set_title("Translation Magnitude (from origin)")
    ax5.grid(True, alpha=0.3)

    # Rotation angle from identity
    ax6 = fig.add_subplot(2, 3, 6)
    angles = [np.degrees(np.linalg.norm(Rotation.from_matrix(p[:3, :3]).as_rotvec())) for p in poses]
    ax6.plot(frames, angles, linewidth=1)
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Angle (deg)")
    ax6.set_title("Rotation Angle (from identity)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VIPE camera-pose trajectory")
    parser.add_argument("--pose", type=Path, default=Path("demo_data/vipe_results/pose/clip_10s_20s.npz"), help="Path to VIPE pose NPZ")
    parser.add_argument("--out", type=Path, default=Path("outputs/vipe_pose_visualization.png"), help="Output image path")
    parser.add_argument("--step", type=int, default=30, help="Frame step for drawing camera axes")
    args = parser.parse_args()

    poses, inds = load_poses(args.pose)
    print(f"Loaded {len(poses)} poses from {args.pose}")
    visualize(poses, inds, args.out, step=args.step)


if __name__ == "__main__":
    main()
