# Dyn-HaMR Input Specification for Hand Track Predictions

**Purpose:** Define the exact output format required from a hand pose estimator (e.g. HaMER, WiLoR) so that Dyn-HaMR can load it directly without running its internal preprocessing pipeline.

**Reference code:**
- `dyn-hamr/data/tools.py` — JSON readers and interpolation logic
- `dyn-hamr/data/dataset.py` — dataset loader and track discovery
- `dyn-hamr/preproc/export_hamer.py` — reference exporter from HaMER pickle outputs

---

## 1. Directory Layout

Dyn-HaMR discovers tracks by listing subdirectories under the `tracks` source path. Each subdirectory name must be a zero-padded 3-digit integer representing the track/hand ID.

```
{track_root}/
├── 000/                          # Track 0 — left hand (is_right = 0)
│   ├── {frame_0001}_mano.json
│   ├── {frame_0001}_keypoints.json
│   ├── {frame_0002}_mano.json
│   ├── {frame_0002}_keypoints.json
│   └── ...
├── 001/                          # Track 1 — right hand (is_right = 1)
│   ├── {frame_0001}_mano.json
│   ├── {frame_0001}_keypoints.json
│   └── ...
└── ...
```

**Rules:**
- Track IDs must match the hand type: `000` → left hand (`is_right = 0`), `001` → right hand (`is_right = 1`).
- Frame filenames inside each track directory must exactly match the extracted image filenames (without extension) plus the suffixes `_mano.json` and `_keypoints.json`.
- Missing frames are allowed; the loader interpolates between existing frames.

---

## 2. MANO Prediction File: `{frame_name}_mano.json`

### 2.1 Schema

```json
{
    "betas": [10 floats],
    "body_pose": [15, 3 floats in angle-axis],
    "global_orient": [3 floats in angle-axis],
    "cam_trans": [3 floats],
    "is_right": 0
}
```

### 2.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `betas` | `(10,)` | `float32` | MANO shape parameters. Must be 10 values. |
| `body_pose` | `(15, 3)` | `float32` | MANO hand pose in angle-axis (Rodrigues) format. 15 joints × 3 axes. **Note:** The name is `body_pose` for historical reasons, but these are hand joints only. |
| `global_orient` | `(3,)` | `float32` | Global wrist orientation in angle-axis format. |
| `cam_trans` | `(3,)` | `float32` | Camera-space translation of the wrist. This is NOT world-space translation; it is relative to the camera frame. |
| `is_right` | scalar | `int` | `0` = left hand, `1` = right hand. Must be constant for all frames within a single track directory. Must match the directory name (`000` → `0`, `001` → `1`). |

### 2.3 Coordinate Systems

- **Rotations:** All rotations (`body_pose`, `global_orient`) must be in **angle-axis (Rodrigues)** format, not rotation matrices or quaternions.
- **Translation:** `cam_trans` is in the camera coordinate system. The Dyn-HaMR optimizer will later refine global translation jointly with camera motion.
- **Units:** Meters (standard MANO convention).

### 2.4 Interpolation Behavior

The loader (`data/tools.py:load_mano_preds`) interpolates missing frames automatically:
- Rotations are interpolated with **Slerp** (spherical linear interpolation) on `SO(3)`.
- Translation and betas are interpolated with **linear interpolation**.
- Interpolation only fills gaps **between the first and last visible frame**; frames before the first detection or after the last detection remain zero-padded.

**Implication:** You do not need to provide a prediction for every frame, but sparse detections will be interpolated.

---

## 3. Keypoint File: `{frame_name}_keypoints.json`

### 3.1 Schema

```json
{
    "people": [
        {
            "pose_keypoints_2d": [J * 3 floats]
        }
    ]
}
```

### 3.2 Field Definitions

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `pose_keypoints_2d` | `(J * 3,)` | `float32` | Flattened array of `[x1, y1, c1, x2, y2, c2, ...]` for `J` joints. `x, y` are in **pixel coordinates** (image space). `c` is a confidence score in `[0, 1]`. |

### 3.3 Coordinate Systems

- **Pixel coordinates:** `x` and `y` must be in the original image pixel coordinate system (not normalized to `[0, 1]`).
- **Confidence:** Any value is acceptable; the loader thresholds at `0.4` internally but then overrides all confidences to `1.0` before optimization.

### 3.4 Interpolation Behavior

The loader (`data/tools.py:load_keypoints_with_interp`) interpolates missing keypoint frames linearly between the first and last valid detection, similar to the MANO parameters.

---

## 4. Important Invariants and Validation

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

## 6. Practical Notes for WiLoR Adaptation

1. **WiLoR outputs rotations as rotation matrices** by default. You must convert them to **angle-axis (Rodrigues)** using `cv2.Rodrigues(R)[0]` before writing to JSON.

2. **WiLoR may predict 23 joints** (full hand) while MANO body_pose expects 15. If your MANO model uses 15 joints, slice or map WiLoR's output accordingly. Check your MANO config — the loader accepts whatever shape is in the JSON, but downstream `BaseSceneModel` may fail if the dimension mismatches the MANO instance.

3. **Camera translation vs. world translation:** WiLoR may output a world-space hand translation. Dyn-HaMR expects `cam_trans` (camera-relative). If WiLoR outputs world-space translation, you must transform it by the camera extrinsics before writing.

4. **Frame naming convention:** The frame name (the part before `_mano.json`) must exactly match the corresponding image filename without extension. Use the same naming for frames, keypoints, and the shot indices JSON.

---

## 7. Example: Minimal Valid Output for 2 Frames, 1 Hand

Directory: `track_preds/demo_seq/000/`

`frame_0001_mano.json`:
```json
{
    "betas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "body_pose": [
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    ],
    "global_orient": [0.0, 0.0, 0.0],
    "cam_trans": [0.0, 0.0, 0.5],
    "is_right": 0
}
```

`frame_0001_keypoints.json`:
```json
{
    "people": [
        {
            "pose_keypoints_2d": [
                100.0, 200.0, 0.9,
                110.0, 210.0, 0.8,
                ...
            ]
        }
    ]
}
```

---

## 8. Checklist for WiLoR Integration

- [ ] For each detected hand, create a `{tid:03d}/` subdirectory (`000` = left, `001` = right).
- [ ] For each frame, write `{frame_name}_mano.json` with keys: `betas`, `body_pose`, `global_orient`, `cam_trans`, `is_right`.
- [ ] For each frame, write `{frame_name}_keypoints.json` with `pose_keypoints_2d` in pixel coordinates.
- [ ] Ensure all rotations are **angle-axis** (not matrices or quaternions).
- [ ] Ensure `body_pose` shape matches your MANO model (typically `(15, 3)`).
- [ ] Ensure `is_right` is `0` for left and `1` for right, and matches the directory name.
- [ ] Ensure frame names match image filenames (without extension).
- [ ] Ensure betas are `(10,)` and translations are `(3,)` in meters.
