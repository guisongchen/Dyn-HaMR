# Dyn-HaMR Left Hand Debug Status

**Date:** 2026-05-16
**Status:** FIXED - Left hand now renders correctly after optimization

## What Was Fixed

1. **WiLoR `cam_trans` scale issue**: Created `fix_wilor_translations.py` to recompute camera translations using actual VIPE intrinsics (~757px) instead of WiLoR's virtual focal length (~37500px). Right hand now renders correctly after optimization.

2. **Left hand `cam_trans[x]` sign bug** (ROOT CAUSE): In `fix_wilor_translations.py`, the x-component of `cam_trans` was computed incorrectly for left hands. Dyn-HaMR's `run_mano()` x-flips joints AFTER the MANO forward pass, meaning:
   - Right hand: `wrist_cam[x] = tx + root_loc[x]`
   - Left hand: `wrist_cam[x] = -tx + root_loc[x]` (due to x-flip)
   
   The original code only handled the right-hand case, causing the left hand to project ~42-203px off target. The fix adds a conditional in `estimate_cam_trans_from_keypoints()`:
   ```python
   if is_right:
       tx = (x_2d - cx) * (root_z + tz) / fx - root_x
   else:
       tx = root_x - (x_2d - cx) * (root_z + tz) / fx
   ```

3. **Dependencies**: Patched `pyrender` for NumPy 2.0 compatibility, installed missing packages, symlinked `_DATA/BMC`.

4. **`init_trans` formula simplified** in `base_scene.py:122-127`: Removed unnecessary `+ root_loc - root_loc` pattern (was no-op with identity camera, but conceptually wrong).

## Verification

After fix, left hand projection errors dropped from 42-203px to <1px. Full optimization ran successfully (50 root + 300 smooth iterations). Videos generated at:
`outputs/logs/video-custom/2026-05-16/dynhamr_out-all-shot-0-0--1/`

## Files Modified / Created

- `dyn-hamr/optim/base_scene.py:122-127` — simplified `init_trans` formula
- `dyn-hamr/fix_wilor_translations.py` — fixed left-hand cam_trans x-component
- `dyn-hamr/optim/losses.py` — depth_constraint added, BMCLoss made conditional
- `dyn-hamr/run_opt.py` — HMP import moved inside `if cfg.run_prior:`
- `dyn-hamr/confs/data/demo_dynhamr.yaml` — created for WiLoR+VIPE pipeline
- `dyn-hamr/diagnose_init.py` — diagnostic script (can be removed)
