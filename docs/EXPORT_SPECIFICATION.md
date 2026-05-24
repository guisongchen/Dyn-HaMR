# Dyn-HaMR Export Format Specification

**Audience:** Developers consuming Dyn-HaMR optimized results.
**Purpose:** Define the canonical output format produced by `run_opt.py` so downstream tools can read it without ambiguity.

Dyn-HaMR writes two files into the output directory:

- `cameras.json` — optimized camera trajectory + intrinsics
- `hands.json` — optimized hand poses (world space)

---

## 1. Camera File: `cameras.json`

```json
{
  "intrins": [fx, fy, cx, cy],
  "height": 1080,
  "width": 1920,
  "num_frames": N,
  "w2c": [[...], ...]
}
```

| Key          | Shape     | Type   | Description |
|--------------|-----------|--------|-------------|
| `intrins`    | [4]       | float  | `[fx, fy, cx, cy]` in pixels. Same for all frames. |
| `height`     | scalar    | int    | Image height in pixels. |
| `width`      | scalar    | int    | Image width in pixels. |
| `num_frames` | scalar    | int    | Total number of frames (N). Must match `len(w2c)`. |
| `w2c`        | [N, 4, 4] | float  | **World-to-camera** (W2C) homogeneous 4×4 matrices. |

Each W2C matrix has the structure `[R | t; 0 0 0 1]`:
- `R[:3,:3]` = world→camera rotation
- `t[:3]` = world→camera translation
- Bottom row = `[0, 0, 0, 1]`

**Note:** These are the *optimized* camera poses. They may differ from the input camera trajectory if camera optimization was enabled.

---

## 2. Hands File: `hands.json`

```json
{
  "hands": [
    {
      "is_right": 1,
      "frames": [
        {
          "frame_id": 0,
          "mano": {
            "betas": [10 floats],
            "body_pose": [45 floats],
            "global_orient": [3 floats],
            "world_trans": [3 floats]
          },
          "keypoints": {
            "pose_keypoints_2d": [63 floats]
          }
        }
      ]
    }
  ]
}
```

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `hands[].is_right` | scalar | int | `1` for right hand, `0` for left hand. |
| `hands[].frames[].frame_id` | scalar | int | Frame index, 0-based, temporally ordered. |
| `hands[].frames[].mano.betas` | [10] | float | MANO shape parameters (shared across all frames for this hand). |
| `hands[].frames[].mano.body_pose` | [45] | float | MANO hand pose (15 joints × 3) in angle-axis format. |
| `hands[].frames[].mano.global_orient` | [3] | float | Wrist **world-space** orientation, angle-axis (Rodrigues). |
| `hands[].frames[].mano.world_trans` | [3] | float | Wrist **world-space** translation `[tx, ty, tz]` in **meters**. |
| `hands[].frames[].keypoints.pose_keypoints_2d` | [63] | float | 21 keypoints × 3 values `[x, y, confidence]`. Copied from the input estimator. Zero-padded if the hand was not detected in this frame. |

---

## 3. Coordinate Conventions

| Frame | Description |
|-------|-------------|
| **World** | Global reference frame. All optimized hand poses (`global_orient`, `world_trans`) live here. |
| **Camera** | Camera-local frame, **+Z = forward (depth)**. Used by `cameras.json` intrinsics for projection. |

Perspective projection with the exported intrinsics:
```
x2d = fx * (Xcam / Zcam) + cx
y2d = fy * (Ycam / Zcam) + cy
```

To project an optimized hand back into 2D:
1. Transform joints from world → camera using `w2c`.
2. Apply the intrinsics formula above.

---

## 4. Differences from Input Format

The export format is intentionally similar to the input format consumed by Dyn-HaMR, but with two critical differences:

| Field | Input (`hands.json`) | Output (`hands.json`) |
|-------|----------------------|-----------------------|
| Translation key | `cam_trans` | `world_trans` |
| Translation frame | Camera space | **World space** |
| `global_orient` frame | Camera space | **World space** |
| `cameras.json` | Input camera poses | **Optimized** camera poses |

**Do not confuse `cam_trans` (input, camera space) with `world_trans` (output, world space).** Downstream tools must use `world_trans` when working with the optimized results.

---

## 5. Validation Checklist

- [ ] `cameras.json` and `hands.json` have the same `num_frames`.
- [ ] `hands[].frames` length matches `num_frames` (zero-padded for missing detections).
- [ ] `world_trans` values are in **meters** and in **world space**, not camera space.
- [ ] `w2c` matrices are valid homogeneous transforms (bottom row `[0,0,0,1]`).
- [ ] Intrinsics are in **pixels**.
