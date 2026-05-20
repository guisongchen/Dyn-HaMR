# Dyn-HaMR Input Specification for WiLoR Hand Track Predictions

**Purpose:** Define the exact output format required from a hand pose estimator (WiLoR) so that Dyn-HaMR can load it directly. Camera poses come from a **separate source** (VIPER, COLMAP, etc.) and are configured independently.

**Reference code:**
- `data/tools.py` — JSON readers and interpolation logic
- `data/dataset.py` — dataset loader and track discovery
- `data/camera_interface.py` — canonical camera data protocol
- `data/camera_loader_vipe.py` — VIPER raw-format camera loader
- `fix_wilor_translations.py` — **required post-processing** to rescale WiLoR's `cam_trans` from virtual focal length (~37500px) to the real camera intrinsics

---

## 1. Pipeline Overview

```
Video → WiLoR (export_dynhamr.py) → fix_wilor_translations.py → Dyn-HaMR
         produces tracks/ + frames/   rescales cam_trans          run_opt.py

VIPER (or other SfM) → pose & intrinsics → Dyn-HaMR camera loader
```

### 1.1 Step 1: WiLoR Export

Run `export_dynhamr.py` from the WiLoR project to produce per-frame MANO predictions and 2D keypoints:
```bash
cd ~/projects/WiLoR
python export_dynhamr.py --video input.mp4 --out_dir dynhamr_out --target_focal_length 757
```

This produces the directory layout described in Section 2.

### 1.2 Step 2: Fix cam_trans Scale

WiLoR outputs `cam_trans` for its internal virtual camera (focal length ~37500px). Dyn-HaMR uses the real camera intrinsics (e.g. VIPER ~757px). Run the fix script to rescale:
```bash
python fix_wilor_translations.py --camera_dir vipe_results --tracks dynhamr_out/tracks/
```

This overwrites `cam_trans` in each `*_mano.json` in-place. **Required before running Dyn-HaMR.**

### 1.3 Step 3: Configure & Run Dyn-HaMR

Edit `confs/data/demo_dynhamr.yaml` to point cameras at the VIPER directory:
```yaml
sources:
  images: ${data.root}/dynhamr_out/frames
  tracks: ${data.root}/dynhamr_out/tracks
camera:
  source: ${data.root}/vipe_results
  type: vipe_pose
```

Then:
```bash
python run_opt.py
```

---

## 2. Directory Layout

```
{track_root}/
├── 000/                          # Track 0 — left hand (is_right = 0)
│   ├── {frame_0000}_mano.json
│   ├── {frame_0000}_keypoints.json
│   ├── {frame_0001}_mano.json
│   ├── {frame_0001}_keypoints.json
│   └── ...
├── 001/                          # Track 1 — right hand (is_right = 1)
│   ├── {frame_0032}_mano.json
│   ├── {frame_0032}_keypoints.json
│   └── ...
└── ...
```

**Rules:**
- Track IDs must match the hand type: `000` → left hand (`is_right = 0`), `001` → right hand (`is_right = 1`).
- Frame filenames inside each track directory must exactly match the extracted image filenames (without extension) plus the suffixes `_mano.json` and `_keypoints.json`.
- Missing frames are allowed; the loader interpolates between existing frames.

---

## 3. MANO Prediction File: `{frame_name}_mano.json`

### 3.1 Schema

```json
{
    "betas": [10 floats],
    "body_pose": [15, 3 floats in angle-axis],
    "global_orient": [3 floats in angle-axis],
    "cam_trans": [3 floats],
    "is_right": 0
}
```

### 3.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `betas` | `(10,)` | `float32` | MANO shape parameters. Must be 10 values. |
| `body_pose` | `(15, 3)` | `float32` | MANO hand pose in angle-axis (Rodrigues) format. 15 joints × 3 axes. The name is `body_pose` for historical reasons — these are hand joints only. |
| `global_orient` | `(3,)` | `float32` | Global wrist orientation in angle-axis format. For left hands, WiLoR predicts on the mirrored image; Dyn-HaMR handles the x-flip internally — do NOT flip this value. |
| `cam_trans` | `(3,)` | `float32` | Camera-space MANO root translation. Must be scaled to the **real camera intrinsics** via `fix_wilor_translations.py` before use. |
| `is_right` | scalar | `int` | `0` = left hand, `1` = right hand. Must be constant for all frames within a single track directory. Must match the directory name (`000` → `0`, `001` → `1`). |

### 3.3 Coordinate Systems

- **Rotations:** All rotations (`body_pose`, `global_orient`) must be in **angle-axis (Rodrigues)** format. WiLoR outputs rotation matrices — `export_dynhamr.py` converts them with `cv2.Rodrigues()`.
- **Translation:** `cam_trans` is in the camera coordinate system. Dyn-HaMR transforms it to world space using the camera extrinsics.
- **Units:** Meters (standard MANO convention).

### 3.4 Interpolation Behavior

The loader (`data/tools.py:load_mano_preds`) interpolates missing frames automatically:
- Rotations are interpolated with **Slerp** (spherical linear interpolation) on `SO(3)`.
- Translation and betas are interpolated with **linear interpolation**.
- Interpolation only fills gaps **between the first and last visible frame**; frames before the first detection or after the last detection remain zero-padded.

---

## 4. Keypoint File: `{frame_name}_keypoints.json`

### 4.1 Schema

```json
{
    "people": [
        {
            "pose_keypoints_2d": [J * 3 floats]
        }
    ]
}
```

### 4.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `pose_keypoints_2d` | `(J * 3,)` | `float32` | Flattened array of `[x1, y1, c1, x2, y2, c2, ...]` for `J` joints. `x, y` are in **pixel coordinates** (image space). `c` is a confidence score in `[0, 1]`. |

### 4.3 Interpolation Behavior

The loader (`data/tools.py:load_keypoints_with_interp`) interpolates missing keypoint frames linearly between the first and last valid detection.

---

## 5. How Dyn-HaMR Uses These Files

### 5.1 Data Loading Flow

```
MultiPeopleDataset.__init__()
  └── scans track_root for subdirectories (000, 001, ...)
      └── checks which frames have _keypoints.json files
          └── filters tracks by length (> MIN_TRACK_LEN)

MultiPeopleDataset.load_data()
  └── for each track:
      ├── load_keypoints_with_interp() → joints2d (T, J, 3)
      └── load_mano_preds() → pose_init, orient_init, trans_init, betas_init, is_right
```

### 5.2 What Becomes `obs_data`

| `obs_data` Key | Source File | Loader Function |
|----------------|-------------|-----------------|
| `joints2d` | `{frame}_keypoints.json` | `load_keypoints_with_interp()` |
| `init_body_pose` | `{frame}_mano.json` → `body_pose` | `load_mano_preds()` |
| `init_body_shape` | `{frame}_mano.json` → `betas` | `load_mano_preds()` |
| `init_root_orient` | `{frame}_mano.json` → `global_orient` | `load_mano_preds()` |
| `init_trans` | `{frame}_mano.json` → `cam_trans` | `load_mano_preds()` |
| `is_right` | `{frame}_mano.json` → `is_right` | `load_mano_preds()` |

These tensors are fed into `BaseSceneModel.initialize(obs_data, cam_data)` to set the initial state for optimization.

---

## 6. Important Invariants and Validation

The Dyn-HaMR dataset loader enforces the following at runtime (`data/dataset.py`):

1. **Track ID ↔ Hand Type Consistency:**
   ```python
   assert int(track_id) == int(is_right[0])
   ```
   Track `000` must contain only left-hand predictions (`is_right = 0`). Track `001` must contain only right-hand predictions (`is_right = 1`).

2. **Hand Type Constancy:**
   ```python
   assert torch.all(is_right == is_right[0])
   ```
   `is_right` must not change across frames within the same track.

3. **Minimum Track Length:**
   Tracks with fewer than `MIN_TRACK_LEN = 60` visible frames are discarded by default when `track_ids: "all"` is used.

---

## 7. Checklist for Integration

- [ ] Run WiLoR `export_dynhamr.py` to produce `tracks/` and `frames/`
- [ ] Run VIPER (or other SfM) to produce camera poses and intrinsics
- [ ] Run `fix_wilor_translations.py --camera_dir <vipe_dir> --tracks <tracks_dir>` to rescale `cam_trans`
- [ ] Set `camera.source` and `camera.type` in the data config YAML
- [ ] Ensure all rotations are **angle-axis** (not matrices or quaternions)
- [ ] Ensure `body_pose` shape matches your MANO model (typically `(15, 3)`)
- [ ] Ensure `is_right` is `0` for left and `1` for right, and matches the directory name
- [ ] Ensure frame names match image filenames (without extension)
- [ ] Ensure betas are `(10,)` and translations are `(3,)` in meters
