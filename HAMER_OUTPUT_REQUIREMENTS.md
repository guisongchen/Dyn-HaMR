# Dyn-HaMR Hand-Track Input Specification

**Audience:** Developers of hand-pose estimators (WiLoR, HaMeR, etc.).  
**Purpose:** Define the output format your tool must produce so that Dyn-HaMR can load hand tracks directly. Camera data is handled separately — see `CAMERA_OUTPUT_REQUIREMENTS.md`.

---

## 1. Directory Layout

Produce one subdirectory per tracked hand, plus a directory of extracted video frames:

```
{output_root}/
├── frames/                           # extracted video frames
│   ├── {frame_0000}.jpg
│   ├── {frame_0001}.jpg
│   └── ...
├── tracks/
│   ├── 000/                          # left hand  (is_right = 0)
│   │   ├── {frame_0000}_mano.json
│   │   ├── {frame_0000}_keypoints.json
│   │   ├── {frame_0001}_mano.json
│   │   ├── {frame_0001}_keypoints.json
│   │   └── ...
│   ├── 001/                          # right hand (is_right = 1)
│   │   ├── {frame_0032}_mano.json
│   │   ├── {frame_0032}_keypoints.json
│   │   └── ...
│   └── ...
```

**Rules:**
- Track directory names **must** match the hand type: `000` = left (`is_right = 0`), `001` = right (`is_right = 1`).
- Frame filenames inside each track directory must exactly match the image filenames in `frames/` (without extension), plus `_mano.json` or `_keypoints.json`.
- Missing frames are allowed — gaps are interpolated, but at least 60 visible frames per track are required.

---

## 2. MANO Prediction File: `{frame_name}_mano.json`

### 2.1 Schema

```json
{
    "betas": [10 floats],
    "body_pose": [15 × 3 floats],
    "global_orient": [3 floats],
    "cam_trans": [3 floats],
    "is_right": 0
}
```

### 2.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `betas` | `(10,)` | float32 | MANO shape parameters. |
| `body_pose` | `(15, 3)` | float32 | Hand pose in **angle-axis (Rodrigues)** format. 15 joints × 3 axes. Named `body_pose` for historical reasons — these are hand joints only. |
| `global_orient` | `(3,)` | float32 | Global wrist orientation in **angle-axis (Rodrigues)** format. For left hands, your estimator likely predicts on a mirrored image — do NOT flip this value; Dyn-HaMR handles x-flip internally. |
| `cam_trans` | `(3,)` | float32 | Camera-space root translation `[tx, ty, tz]` in **meters**. Must be scaled to the **real camera intrinsics** (not your estimator's virtual focal length). See Section 4. |
| `is_right` | scalar | int | `0` = left hand, `1` = right hand. Must be constant for all frames within a single track directory. |

### 2.3 Coordinate Conventions

- **Rotations:** Angle-axis (Rodrigues) vectors, `(3,)` per joint. Magnitude = radians, direction = rotation axis.
- **Translation:** `cam_trans` is in camera space (meters). `tz` = depth from camera. Dyn-HaMR transforms it to world space using camera extrinsics.
- **Left-hand x-flip:** If your estimator works on horizontally mirrored frames for left hands, output `global_orient` and `body_pose` as predicted. Dyn-HaMR applies the x-flip internally during the MANO forward pass.

---

## 3. Keypoint File: `{frame_name}_keypoints.json`

### 3.1 Schema

```json
{
    "people": [
        {
            "pose_keypoints_2d": [J × 3 floats]
        }
    ]
}
```

### 3.2 Field Definition

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `pose_keypoints_2d` | `(J×3,)` | float32 | Flattened `[x1, y1, c1, x2, y2, c2, ...]`. `x, y` in **pixel coordinates** (image space). `c` ∈ [0, 1] confidence. |

---

## 4. Post-Processing: Fix `cam_trans` Scale

Most hand-pose estimators output `cam_trans` for a **virtual** camera (focal length ~3000–40000 px). Dyn-HaMR uses the **real** camera intrinsics (from SfM). The two must match, or the hand will project to the wrong location.

Run the provided rescaling script after producing tracks and camera data:

```bash
python fix_wilor_translations.py --camera_dir <vipe_dir> --tracks <tracks_dir>
```

This adjusts `cam_trans` in every `*_mano.json` so the MANO wrist projects correctly onto the detected 2D keypoint under the real camera. **Run this before starting Dyn-HaMR.**

---

## 5. Invariants and Validation

Your output must satisfy these rules (enforced at runtime):

1. **Track ID = hand type**: Directory `000` must contain only `is_right = 0`. Directory `001` must contain only `is_right = 1`. If your estimator detects both hands, use separate directories per hand.

2. **Hand type constancy**: `is_right` must be the same value in every `*_mano.json` within the same track directory.

3. **Minimum track length**: At least 60 frames with valid keypoint detections per track. Shorter tracks are skipped.

4. **Frame names match**: The filename stem (without extension) of each `.jpg` in `frames/` must exactly match the stem used in `_mano.json` and `_keypoints.json`.

---

## 6. Validation Checklist

- [ ] Output directory has `frames/` and `tracks/` subdirectories.
- [ ] Track directories are named `000`, `001`, … matching hand type.
- [ ] All rotations (`body_pose`, `global_orient`) are **angle-axis** — not rotation matrices, not quaternions.
- [ ] `body_pose` shape is `(15, 3)` per frame.
- [ ] `betas` shape is `(10,)`.
- [ ] `cam_trans` shape is `(3,)` in meters.
- [ ] `is_right` is `0` or `1` and matches the directory name.
- [ ] Frame name stems match between `frames/` and `tracks/`.
- [ ] Ran `fix_wilor_translations.py` (or equivalent) to rescale `cam_trans` to real camera intrinsics.
