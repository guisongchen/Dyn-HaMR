# Dyn-HaMR Camera Input Specification

**Purpose:** Define the exact format that external camera estimators (VIPE, COLMAP, etc.) must produce so that Dyn-HaMR can load camera poses and intrinsics directly. Camera data is configured **separately** from hand-tracking data — see `confs/data/demo_dynhamr.yaml`.

**Reference code:**
- `data/camera_interface.py` — canonical protocol every loader must satisfy
- `data/camera_loader_vipe.py` — reference loader for VIPE raw format
- `data/dataset.py:load_camera_data()` — loader dispatch by `camera.type`

---

## 1. Config Entry

The data config YAML must include a `camera` section:

```yaml
camera:
  source: /path/to/camera_output
  type: vipe_pose          # or canonical_npz, colmap, ...
```

Dyn-HaMR picks the correct loader based on `type`. The `source` path is passed directly to that loader.

---

## 2. Supported Camera Types

### 2.1 `vipe_pose` — VIPE Raw Output

The source directory must contain exactly two subdirectories:

```
{source}/
├── pose/
│   └── *.npz                # first .npz found is used
└── intrinsics/
    └── *.npz                # first .npz found is used
```

#### pose/*.npz

| Key    | Shape   | Dtype     | Description |
|--------|---------|-----------|-------------|
| `data` | (N, 4, 4) | float32 | Camera-to-world (C2W) homogeneous matrices. Dyn-HaMR inverts them to W2C. |

Each 4×4 matrix encodes `[R ∣ t; 0 0 0 1]` where `R` (3×3) is the rotation and `t` (3×1) is the translation of the camera in world space.

#### intrinsics/*.npz

| Key    | Shape  | Dtype     | Description |
|--------|--------|-----------|-------------|
| `data` | (N, 4) | float32  | `[fx, fy, cx, cy]` per frame |

- `fx`, `fy` — focal length in pixels (same for all frames if camera has fixed lens).
- `cx`, `cy` — principal point in pixels (typically image-width/2, image-height/2).
- **N must equal the number of frames** in `pose/*.npz`.

### 2.2 `canonical_npz` — Pre-converted `.npz` File

The source path is expected to be a directory containing a single `cameras.npz` file:

| Key       | Shape   | Dtype   | Description |
|-----------|---------|---------|-------------|
| `w2c`     | (N, 4, 4) | float32 | World-to-camera (W2C) homogeneous matrices |
| `intrins` | (N, 4)    | float32 | `[fx, fy, cx, cy]` per frame |
| `height`  | scalar    | int     | Image height in pixels |
| `width`   | scalar    | int     | Image width in pixels |
| `focal`   | scalar    | float   | Nominal focal length (used as default if `intrins` absent) |

Each 4×4 matrix encodes `[R ∣ t; 0 0 0 1]` where `R = w2c[:3, :3]` is the **world-to-camera rotation** and `t = w2c[:3, 3]` is the **world-to-camera translation**.

### 2.3 Adding a New Camera Type

1. Create a loader class that satisfies `CameraDataProtocol` (see `data/camera_interface.py`):

```python
class CameraDataProtocol(Protocol):
    cam_R: torch.Tensor   # (T, 3, 3)  world-to-camera rotation
    cam_t: torch.Tensor   # (T, 3)     world-to-camera translation
    intrins: torch.Tensor # (T, 4)     [fx, fy, cx, cy]
    is_static: bool

    def world2cam(self): ...   # return cam_R, cam_t
    def cam2world(self): ...   # return inverse transform
    def as_dict(self) -> dict: ...  # canonical dict for downstream
```

2. The class `__init__` signature must match:
   ```
   __init__(self, source_dir, seq_len, img_size, is_static,
            data_interval=(0, -1), track_interval=(0, -1))
   ```

3. Register the new `type` string in `data/dataset.py:load_camera_data()`.

---

## 3. Coordinate Conventions

| Frame       | Representation          | Description |
|-------------|-------------------------|-------------|
| World       | `cam2world()`           | Global reference frame. MANO poses are optimized in this frame. |
| Camera      | `cam_R @ point + cam_t` | Camera local frame. `cam_R` and `cam_t` define the world→camera transform. Depth is `+Z`. |

- **`cam_R`**: world-to-camera rotation matrix, shape (T, 3, 3).
- **`cam_t`**: world-to-camera translation vector, shape (T, 3).

Dyn-HaMR uses these to:
1. Transform initial HaMeR/WiLoR predictions from camera space to world space (via `cam2world()`).
2. Project optimized world-space joints back to 2D for the reprojection loss (via `world2cam()`).

---

## 4. Static vs Dynamic Cameras

Set `is_static = True` when all cameras share the same viewpoint (e.g., tripod-mounted). When cameras are static, Dyn-HaMR disables world-scale optimization because translation depth and object scale are inherently ambiguous from a single viewpoint.

---

## 5. Example: VIPE → Dyn-HaMR Flow

```
VIPE → vipe_results/pose/clip_10s_20s.npz      (C2W matrices, key: data)
     → vipe_results/intrinsics/clip_10s_20s.npz  (fx,fy,cx,cy, key: data)

Config:
  camera:
    source: /path/to/vipe_results
    type: vipe_pose

Dyn-HaMR → VIPECameraData → inverts C2W → provides cam_R, cam_t, intrins to optimizer
```

No manual conversion step is required.
