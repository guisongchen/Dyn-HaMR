# Dyn-HaMR Camera Input Specification

**Audience:** Developers of external camera-pose estimators (VIPE, COLMAP, etc.).  
**Purpose:** Define the output format your tool must produce so that Dyn-HaMR can consume it directly — no intermediate conversion.

Dyn-HaMR expects camera data in one of the formats below. Choose whichever is easiest for your estimator.

---

## 1. Format A: VIPE Layout (recommended)

Produce two files in separate subdirectories. Dyn-HaMR discovers them by extension — file name does not matter.

```
{output_dir}/
├── pose/
│   └── *.npz
└── intrinsics/
    └── *.npz
```

### pose/*.npz

| Key    | Shape   | Dtype     | Description |
|--------|---------|-----------|-------------|
| `data` | (N, 4, 4) | float32 | **Camera-to-world** (C2W) homogeneous 4×4 matrices. Dyn-HaMR inverts them internally. |

Each matrix has the structure `[R ∣ t; 0 0 0 1]`:
- `R` = 3×3 rotation matrix (top-left block)
- `t` = 3×1 translation vector (top-right column)
- Bottom row: `[0, 0, 0, 1]`

These encode the camera's position and orientation **in world space** (camera→world transform).

### intrinsics/*.npz

| Key    | Shape  | Dtype     | Description |
|--------|--------|-----------|-------------|
| `data` | (N, 4) | float32  | `[fx, fy, cx, cy]` per frame, in pixels. |

- **N must equal** the number of frames in `pose/*.npz`.
- `(fx, fy)` = focal lengths. `(cx, cy)` = principal point (usually image-width/2, image-height/2).
- All frames may share identical intrinsics if the camera lens is fixed.

---

## 2. Format B: Canonical `.npz`

Produce a single file named `cameras.npz`:

| Key       | Shape     | Dtype   | Description |
|-----------|-----------|---------|-------------|
| `w2c`     | (N, 4, 4)  | float32 | **World-to-camera** (W2C) homogeneous 4×4 matrices |
| `intrins` | (N, 4)     | float32 | `[fx, fy, cx, cy]` per frame, in pixels |
| `height`  | scalar     | int     | Image height in pixels |
| `width`   | scalar     | int     | Image width in pixels |
| `focal`   | scalar     | float   | Nominal focal length (used as fallback only) |

Unlike Format A, these matrices encode the transform from **world space into camera space**. Each 4×4 has the structure `[R ∣ t; 0 0 0 1]` where `R[:3,:3]` is world→camera rotation and `t[:3]` is world→camera translation.

---

## 3. Which Format to Use

| If your estimator outputs… | Use |
|---------------------------|-----|
| Separate pose + intrinsics arrays in C2W convention | Format A (VIPE) |
| A single self-contained file in W2C convention | Format B (canonical) |

Dyn-HaMR's data config YAML selects the format:
```yaml
camera:
  source: /path/to/your/output_dir
  type: vipe_pose     # Format A
# type: canonical_npz # Format B
```

---

## 4. Coordinate Conventions

| Frame       | Description |
|-------------|-------------|
| World       | Global reference frame. MANO hand poses live here. |
| Camera      | Camera-local frame, **+Z = forward (depth)**. |

**Format A (VIPE):** you provide C2W. Dyn-HaMR computes W2C = inv(C2W).  
**Format B (canonical):** you provide W2C directly.

Dyn-HaMR uses these transforms for two purposes:
1. Convert initial hand predictions from camera space into world space.
2. Project optimized world-space joints back to 2D pixels for the reprojection loss.

The intrinsics define perspective projection:
```
x2d = fx * (Xcam / Zcam) + cx
y2d = fy * (Ycam / Zcam) + cy
```

---

## 5. Static vs Moving Cameras

If all cameras share the same viewpoint (e.g. tripod-mounted), your loader should mark `is_static = True`. Dyn-HaMR then disables world-scale optimization, because depth and object scale are ambiguous from a single viewpoint.

Static cameras also work correctly — Dyn-HaMR's reprojection loss handles them identically.

---

## 6. Validation Checklist

- [ ] Pose and intrinsics arrays have the **same number of frames (N)**.
- [ ] N matches or exceeds the video frame count in Dyn-HaMR's image directory.
- [ ] Camera poses are temporally ordered (frame i corresponds to frame i of the video).
- [ ] All intrinsics values are in **pixels** (not mm, not normalized).
- [ ] Rotation matrices are orthonormal (no scaling or shearing artifacts).
