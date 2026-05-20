import os
import glob
import numpy as np
import torch

from .camera_interface import CameraDataProtocol


class VIPECameraData(CameraDataProtocol):
    """
    Loads camera poses from raw VIPER output files.

    Expected directory layout::

        {source}/
          pose/clip_*.npz         →  ``data`` (N, 4, 4)  camera-to-world matrices
          intrinsics/clip_*.npz   →  ``data`` (N, 4)   [fx, fy, cx, cy]

    VIPER stores **camera-to-world** (C2W) poses; this loader computes the
    exact matrix inverse to obtain the canonical **world-to-camera** (W2C)
    representation consumed by the optimisation pipeline.
    """

    def __init__(self, source_dir, seq_len, img_size, is_static,
                 data_interval=(0, -1), track_interval=(0, -1)):
        self.is_static = is_static

        data_start, data_end = data_interval
        if data_end < 0:
            data_end += seq_len + 1
        data_len = data_end - data_start

        sidx, eidx = track_interval
        if eidx < 0:
            eidx += data_len + 1

        sidx += data_start
        eidx += data_start
        T = eidx - sidx

        pose_dir = os.path.join(source_dir, "pose")
        intr_dir = os.path.join(source_dir, "intrinsics")

        poses_npz = _find_single_npz(pose_dir)
        intr_npz = _find_single_npz(intr_dir)
        if poses_npz is None or intr_npz is None:
            _init_default(self, img_size, T)
            return

        # VIPER stores camera-to-world (C2W); invert to world-to-camera (W2C)
        vip_c2w = np.load(poses_npz)["data"]          # (N, 4, 4)
        vip_intr = np.load(intr_npz)["data"]          # (N, 4)

        if len(vip_c2w) != len(vip_intr):
            raise ValueError(
                f"VIPER pose count ({len(vip_c2w)}) != intrinsics count ({len(vip_intr)})"
            )
        if eidx > len(vip_c2w):
            raise ValueError(
                f"Requested slice [{sidx}:{eidx}] > VIPER data ({len(vip_c2w)})"
            )

        w2c = np.linalg.inv(vip_c2w[sidx:eidx])      # (T, 4, 4)
        self.cam_R = torch.from_numpy(w2c[:, :3, :3]).float()
        self.cam_t = torch.from_numpy(w2c[:, :3, 3]).float()

        img_w, img_h = img_size
        canon_w = int(vip_intr[0, 2] * 2)
        scale = img_w / canon_w if canon_w > 0 else 1.0
        self.intrins = torch.from_numpy(vip_intr[sidx:eidx] * scale).float()


def _find_single_npz(directory):
    candidates = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not candidates:
        print(f"WARNING: No .npz found in {directory}, using default cameras")
        return None
    return candidates[0]


def _init_default(obj, img_size, seq_len):
    w, h = img_size
    default_focal = 0.5 * (w + h)
    obj.intrins = torch.tensor(
        [default_focal, default_focal, w / 2, h / 2]
    )[None].repeat(seq_len, 1)
    obj.cam_R = torch.eye(3)[None].repeat(seq_len, 1, 1)
    obj.cam_t = torch.zeros(seq_len, 3)
