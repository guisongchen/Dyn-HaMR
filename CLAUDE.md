# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dyn-HaMR reconstructs 4D global hand motion from monocular videos recorded by dynamic cameras. It is a CVPR 2025 Highlight project from Imperial College London.

The codebase is a PyTorch optimization pipeline that takes 2D hand detections + initial MANO pose estimates + camera trajectory, and jointly optimizes hand poses, shapes, and camera parameters over time.

## Environment

The project requires a conda environment because MANO model loading depends on `chumpy`, which is incompatible with Python 3.13+ in the base environment. Use the `hamer` conda env:

```bash
conda run -n hamer python <script>.py
```

To create the environment from scratch, see `install_conda.sh` or `install_pip.sh` in the project root (if present). Model checkpoints are fetched via `prepare.sh`.

## Key Commands

### Run optimization
```bash
# Video input with DROID-SLAM camera estimation
python run_opt.py data=video run_opt=True data.seq=demo1 is_static=False

# Static camera
python run_opt.py data=video run_opt=True data.seq=demo1 is_static=True

# Optimization + visualization in one step
python -u run_opt.py data=video run_opt=True run_vis=True is_static=False
```

### Evaluate results
```bash
# Diagnostic comparison (input vs optimized)
python eval_opt.py --input_hands demo_data/hands.json --opt_hands outputs/hands.json --opt_cameras outputs/cameras.json --out_dir outputs/eval

# Reprojection error analysis (single summary figure)
python eval_reproj.py --hands outputs/hands.json --cameras outputs/cameras.json --out_dir outputs/reproj_eval

# Per-frame overlay visualization
python vis_reproj.py --input_hands demo_data/hands.json --opt_hands outputs/hands.json --opt_cameras outputs/cameras.json --img_dir demo_data/frames --out_dir outputs/reproj_vis --write_video
```

### Render visualization
```bash
python run_vis.py --log_root outputs/
```

## Architecture

### Config System
Configs are loaded with `OmegaConf` and merged in `run_opt.py::load_config()`:
- `confs/config.yaml` — Main config (model flags, paths, MANO, vis phases)
- `confs/data/<data>.yaml` — Data source config (video paths, frame ranges)
- `confs/optim.yaml` — Optimization hyperparameters, loss weights per stage

Loss weights in `optim.yaml` are arrays of 3 values corresponding to the 3 optimization stages.

### Optimization Pipeline (`optim/`)

The pipeline runs in stages using LBFGS with strong Wolfe line search:

1. **RootOptimizer** (`optim/optimizers.py`): Fits root orientation and translation per frame independently.
2. **SmoothOptimizer** (`optim/optimizers.py`): Fits full MANO poses and betas with temporal smoothness constraints.
3. **Motion Prior** (optional, `HMP/fitting.py`): Activated via `run_prior=True`. Requires chunk size 128.

`BaseSceneModel` (`optim/base_scene.py`) is the central `nn.Module` that holds:
- MANO body model
- `CameraParams` (`optim/params.py`) for camera variables
- Forward pass computes losses (joints2d, depth, smoothness, bone length, priors, etc.)

Losses are defined in `optim/losses.py` and `optim/bio_loss.py`.

### Body Model (`body_model/`)

- `body_model/mano_wrapper.py`: Extends `MANOLayer` with extra fingertip joints and OpenPose joint reordering. The `mano_to_openpose` mapping is `[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]`.
- `body_model/utils.py`: `run_mano()` handles batched forward passes and automatically mirrors x-coordinates for left hands (`is_right=0`).

### Data Pipeline (`data/`)

- `data/dataset.py`: `MultiPeopleDataset` loads the canonical input format (`hands.json` + `cameras.json` + frames).
- Input `hands.json` uses `cam_trans` (camera space). The dataset converts these to world space internally using the camera matrices.
- `data/tools.py`: Loading utilities with interpolation for missing frames.

### Camera and Geometry (`geometry/`)

- `geometry/camera.py`: `reproject()` projects world-space 3D points to 2D using camera extrinsics and intrinsics.
- Camera convention: **+Z = forward (depth)**.
- `cameras.json` stores `w2c` (world-to-camera) 4×4 matrices, not camera-to-world.

### Visualization (`vis/`)

- `vis/viewer.py`: Pyrender-based offscreen viewer for mesh rendering.
- `vis/output.py`: Video generation, grid composition, and result preparation.
- `vis/tools.py`: OpenCV keypoint drawing utilities.

## Export Format

`run_opt.py` exports two canonical JSON files after optimization:

- **`cameras.json`**: `w2c` [T,4,4], `intrins` [fx,fy,cx,cy], `num_frames`
- **`hands.json`**: Optimized MANO params in **world space** (`world_trans`, `global_orient`), plus `pose_keypoints_2d` and `pose_keypoints_3d`

Critical convention: Output uses `world_trans` (world space), while input uses `cam_trans` (camera space). Do not confuse them.

The full specification is in `docs/EXPORT_SPECIFICATION.md`. Input requirements for hand trackers are in `docs/HAMER_OUTPUT_REQUIREMENTS.md` and `docs/CAMERA_OUTPUT_REQUIREMENTS.md`.

## Code Conventions

- `is_right`: `0` = Left hand, `1` = Right hand.
- Batch dimensions: `B` = number of hands/tracks, `T` = number of frames.
- MANO pose body has 15 joints (45-D in angle-axis). With wrist, full hand is 16 joints; the wrapper appends 5 fingertips for 21 joints total.
- When running MANO forward in `eval_opt.py` or `vis_reproj.py`, the model is initialized as right-hand only (`is_rhand=True`); left hands are handled by flipping the x-coordinate post-forward.
