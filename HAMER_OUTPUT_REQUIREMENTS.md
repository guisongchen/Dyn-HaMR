# Dyn-HaMR Hand-Track Input Specification

**Audience:** Developers of hand-pose estimators (WiLoR, HaMeR, etc.).  
**Purpose:** Define the output format your tool must produce so that Dyn-HaMR can load hand tracks directly. Camera data is handled separately — see `CAMERA_OUTPUT_REQUIREMENTS.md`.

---

## 1. Output

Produce **a single JSON file** containing all hand predictions. Do NOT export video frames — Dyn-HaMR extracts frames directly from the source video internally.

```
{output_root}/
├── hands.json
```

---

## 2. Hands File: `hands.json`

A single JSON file containing an array of hands. Each entry carries `is_right` and all its frame predictions.

### 2.1 Schema

```json
{
    "hands": [
        {
            "is_right": 0,
            "frames": [
                {
                    "frame_id": 0,
                    "mano": {
                        "betas": [10 floats],
                        "body_pose": [15 × 3 floats],
                        "global_orient": [3 floats],
                        "cam_trans": [3 floats]
                    },
                    "keypoints": {
                        "pose_keypoints_2d": [J × 3 floats]
                    }
                }
            ]
        },
        {
            "is_right": 1,
            "frames": [
                {
                    "frame_id": 32,
                    "mano": {
                        "betas": [10 floats],
                        "body_pose": [15 × 3 floats],
                        "global_orient": [3 floats],
                        "cam_trans": [3 floats]
                    },
                    "keypoints": {
                        "pose_keypoints_2d": [J × 3 floats]
                    }
                }
            ]
        }
    ]
}
```

### 2.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `hands` | list | — | Ordered list of hand entries. |
| `hands[].is_right` | scalar | int | `0` = left hand, `1` = right hand. |
| `hands[].frames` | list | — | Per-frame predictions, ordered by ascending `frame_id`. |
| `hands[].frames[].frame_id` | scalar | int | Frame index in the source video (0-based). |
| `hands[].frames[].mano.betas` | `(10,)` | float32 | MANO shape parameters. |
| `hands[].frames[].mano.body_pose` | `(15, 3)` | float32 | Hand pose in **angle-axis (Rodrigues)** format. 15 joints × 3 axes. Named `body_pose` for historical reasons — these are hand joints only. |
| `hands[].frames[].mano.global_orient` | `(3,)` | float32 | Global wrist orientation in **angle-axis (Rodrigues)** format. For left hands, your estimator likely predicts on a mirrored image — do NOT flip this value; Dyn-HaMR handles x-flip internally. |
| `hands[].frames[].mano.cam_trans` | `(3,)` | float32 | Camera-space root translation `[tx, ty, tz]` in **meters**. Must be scaled to the **real camera intrinsics** (not your estimator's virtual focal length). See Section 4. |
| `hands[].frames[].keypoints.pose_keypoints_2d` | `(J×3,)` | float32 | Flattened `[x1, y1, c1, x2, y2, c2, ...]`. `x, y` in **pixel coordinates** (image space). `c` ∈ [0, 1] confidence. |

### 2.3 Coordinate Conventions

- **Rotations:** Angle-axis (Rodrigues) vectors, `(3,)` per joint. Magnitude = radians, direction = rotation axis.
- **Translation:** `mano.cam_trans` is in camera space (meters). `tz` = depth from camera. Dyn-HaMR transforms it to world space using camera extrinsics.
- **Left-hand x-flip:** If your estimator works on horizontally mirrored frames for left hands, output `mano.global_orient` and `mano.body_pose` as predicted. Dyn-HaMR applies the x-flip internally during the MANO forward pass.

---

## 3. Post-Processing: Fix `cam_trans` Scale

**Pre-requisite:** You must know the real camera intrinsics (`fx, fy, cx, cy` in pixels) before exporting hand-pose estimates. At a minimum, you need the focal length `fx`. These come from your camera-pose estimator's output — see `CAMERA_OUTPUT_REQUIREMENTS.md` for the expected format. Without them, `cam_trans` cannot be rescaled and the hand will project incorrectly.

Most hand-pose estimators output `cam_trans` for a **virtual** camera (focal length ~3000–40000 px). Dyn-HaMR uses the **real** camera intrinsics (from SfM). The two must match, or the hand will project to the wrong location.

Run the provided rescaling script after producing tracks and camera data:

```bash
python fix_wilor_translations.py --camera_dir <vipe_dir> --hands <hands_json>
```

This adjusts `mano.cam_trans` in every frame so the MANO wrist projects correctly onto the detected 2D keypoint under the real camera. **Run this before starting Dyn-HaMR.**

---

## 4. Invariants and Validation

Your output must satisfy these rules (enforced at runtime):

1. **`is_right` discriminates hand type**: `is_right = 0` for left hands, `is_right = 1` for right hands.
2. **Minimum hand track length**: At least 60 frames with valid keypoint detections per hand entry. Shorter entries are skipped.

---

## 5. Validation Checklist

- [ ] Output is a single `hands.json` file.
- [ ] Top-level `hands` array contains one entry per tracked hand.
- [ ] Each entry has `is_right` (`0` or `1`) and a `frames` array.
- [ ] All rotations (`mano.body_pose`, `mano.global_orient`) are **angle-axis** — not rotation matrices, not quaternions.
- [ ] `mano.body_pose` shape is `(15, 3)` per frame.
- [ ] `mano.betas` shape is `(10,)`.
- [ ] `mano.cam_trans` shape is `(3,)` in meters.
- [ ] Ran `fix_wilor_translations.py` (or equivalent) to rescale `mano.cam_trans` to real camera intrinsics.
