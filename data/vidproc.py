import os


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def verify_frames(img_dir):
    if not is_nonempty(img_dir):
        raise FileNotFoundError(
            f"Frames not found at {img_dir}. Place extracted frames in this directory."
        )


def verify_tracks(track_dir):
    if not is_nonempty(track_dir):
        raise FileNotFoundError(
            f"Tracks not found at {track_dir}. Place per-frame MANO predictions and "
            "keypoints in this directory."
        )


def verify_cameras(cam_dir):
    if not is_nonempty(cam_dir):
        raise FileNotFoundError(
            f"Cameras not found at {cam_dir}. Place cameras.npz in this directory."
        )
