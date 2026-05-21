# Dyn-HaMR Camera Input Specification

**Audience:** Developers of external camera-pose estimators (COLMAP, etc.).
**Purpose:** Define the output format your tool must produce so that Dyn-HaMR can consume it directly — no intermediate conversion.

Dyn-HaMR expects a single `cameras.npz` file.

---

## Canonical Format: `cameras.npz`

| Key       | Shape     | Dtype   | Description |
|-----------|-----------|---------|-------------|
| `w2c`     | (N, 4, 4)  | float32 | **World-to-camera** (W2C) homogeneous 4×4 matrices |
| `intrins` | (N, 4)     | float32 | `[fx, fy, cx, cy]` per frame, in pixels |
| `height`  | scalar     | int     | Image height in pixels |
| `width`   | scalar     | int     | Image width in pixels |
| `focal`   | scalar     | float   | Nominal focal length (used as fallback only) |

Each W2C 4×4 has the structure `[R ∣ t; 0 0 0 1]`:
- `R[:3,:3]` = world→camera rotation
- `t[:3]` = world→camera translation
- Bottom row: `[0, 0, 0, 1]`

Dyn-HaMR inverts these internally to transform hand poses from camera space into world space.

---

## Configuration

```yaml
camera:
  source: /path/to/your/output_dir   # directory containing cameras.npz
  type: canonical_npz
```

---

## Coordinate Conventions

| Frame  | Description |
|--------|-------------|
| World  | Global reference frame. MANO hand poses live here. |
| Camera | Camera-local frame, **+Z = forward (depth)**. |

The intrinsics define perspective projection:
```
x2d = fx * (Xcam / Zcam) + cx
y2d = fy * (Ycam / Zcam) + cy
```

---

## Static vs Moving Cameras

If all cameras share the same viewpoint (e.g. tripod-mounted), set `is_static = True`. Dyn-HaMR then disables world-scale optimization, because depth and object scale are ambiguous from a single viewpoint.

---

## Validation Checklist

- [ ] Pose and intrinsics arrays have the **same number of frames (N)**.
- [ ] N matches or exceeds the video frame count in Dyn-HaMR's image directory.
- [ ] Camera poses are temporally ordered (frame i corresponds to frame i of the video).
- [ ] All intrinsics values are in **pixels** (not mm, not normalized).
- [ ] Rotation matrices are orthonormal (no scaling or shearing artifacts).
