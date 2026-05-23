import os
import tempfile
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from body_model import MANO

from data import get_dataset_from_cfg
from optim.output import (
    get_results_paths,
    load_result,
    save_input_poses,
)
from util.tensor import get_device, move_to, detach_all, to_torch
from vis.output import prep_result_vis, animate_scene, make_video_grid_2x2
from vis.tools import vis_keypoints
from vis.viewer import init_viewer


def run_vis(
    cfg,
    dataset,
    out_dir,
    dev_id,
    phases=["smooth_fit"],
    render_views=["src_cam", "above", "side"],
    make_grid=True,
    overwrite=False,
    save_dir=None,
    render_kps=False,
    render_layers=False,
    save_frames=False,
    **kwargs
):
    save_dir = out_dir if save_dir is None else save_dir

    if render_kps:
        render_keypoints_2d(dataset, save_dir, overwrite=overwrite)

    if len(render_views) < 1:
        return

    out_ext = "/" if render_layers or save_frames else ".mp4"
    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        res_dir = os.path.join(out_dir, phase)
        if phase == "input":
            res = get_input_dict(dataset)
            it = f"{0:06d}"

        elif os.path.isdir(res_dir):
            res_path_dict = get_results_paths(res_dir)
            it = sorted(res_path_dict.keys())[-1]
            res = load_result(res_path_dict[it])["world"]

        else:
            print(f"{res_dir} does not exist, skipping")
            continue

        out_name = f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}"
        phase_max_iters[phase] = it

        out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        if not overwrite and all(os.path.exists(p) for p in out_paths):
            print("FOUND OUT PATHS", out_paths)
            continue

        phase_results[phase] = out_name, res

    if len(phase_results) > 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_names, res_dicts = zip(*phase_results.values())
            tmp_names = [f"{tmpdir}/{dataset.seq_name}_{phase}_final_{it}" for phase, it in phase_max_iters.items()]
            render_results(
                cfg,
                dataset,
                res_dicts,
                tmp_names,
                render_views=render_views,
                render_layers=render_layers,
                save_frames=save_frames,
                **kwargs,
            )

            if make_grid:
                for phase, it in phase_max_iters.items():
                    grid_path = f"{save_dir}/{dataset.seq_name}_{phase}_grid.mp4"
                    vid_paths = [
                        f"{tmpdir}/{dataset.seq_name}_{phase}_final_{it}_src_cam.mp4",
                        f"{tmpdir}/{dataset.seq_name}_{phase}_final_{it}_front.mp4",
                        f"{tmpdir}/{dataset.seq_name}_{phase}_final_{it}_above.mp4",
                        f"{tmpdir}/{dataset.seq_name}_{phase}_final_{it}_side.mp4",
                    ]
                    make_video_grid_2x2(
                        grid_path,
                        vid_paths,
                        fps=cfg.fps,
                        overwrite=True,
                    )


def get_input_dict(dataset):
    dataset.load_data(interp_input=False)
    d = dataset.data_dict
    input_params = {
        "pose_body": np.stack(d["init_body_pose"], axis=0),
        "trans": np.stack(d["init_trans"], axis=0),
        "root_orient": np.stack(d["init_root_orient"], axis=0),
    }
    input_params = to_torch(input_params)
    return input_params


def render_keypoints_2d(dataset, save_dir, overwrite=False):
    """
    render 2d keypoints for each track
    """
    dataset.load_data()
    out_dir = f"{save_dir}/{dataset.seq_name}_joints2d"
    B, T = dataset.n_tracks, dataset.seq_len
    if not overwrite and (os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= B * T):
        print(f"Keypoints already rendered in {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for i, tid in enumerate(dataset.track_ids):
        joints2d = dataset.data_dict["joints2d"][i]  # (T, J, 3)
        for t, sel_img_name in enumerate(dataset.sel_img_names):
            img = vis_keypoints(joints2d[t : t + 1], dataset.img_size)
            out_path = f"{out_dir}/{sel_img_name}_{tid}.png"
            imageio.imwrite(out_path, img)



def _set_pred_keypoints(vis, hand_model, res_dict):
    from geometry import camera as cam_util
    from body_model import run_mano

    joints3d_op = run_mano(
        hand_model,
        res_dict["trans"],
        res_dict["root_orient"],
        res_dict["pose_body"],
        res_dict["is_right"],
        res_dict.get("betas", None),
    )["joints"]

    cam_R = res_dict["cam_R"]
    cam_t = res_dict["cam_t"]
    intrins = res_dict["intrins"]

    if intrins.ndim == 1:
        intrins = intrins[None, None].expand(cam_R.shape[0], cam_R.shape[1], -1)
    elif intrins.ndim == 2:
        intrins = intrins[None].expand(cam_R.shape[0], -1, -1)

    joints2d_pred = cam_util.reproject(
        joints3d_op, cam_R, cam_t, intrins[0, :, :2], intrins[0, :, 2:]
    )
    vis.set_pred_keypoints_seq(
        [joints2d_pred[0, t].cpu().numpy() for t in range(joints2d_pred.shape[1])]
    )


def render_results(cfg, dataset, res_dicts, out_names, **kwargs):
    assert len(res_dicts) == len(out_names)
    if len(res_dicts) < 1:
        print("no results to render, skipping")
        return

    B = len(dataset)
    T = dataset.seq_len
    device = get_device()
    obs_data = move_to(next(iter(DataLoader(dataset, batch_size=B, shuffle=False))), device)
    cam_data = dataset.get_camera_data()

    cfg.paths.MANO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mano")
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    hand_model = MANO(batch_size=B * T, pose2rot=True, **mano_cfg).to(device)

    vis = init_viewer(
        dataset.img_size,
        cam_data["intrins"][0],
        vis_scale=1.0,
        bg_paths=dataset.sel_img_paths,
        fps=cfg.fps,
    )

    if kwargs.get('render_keypoints', False):
        dataset.load_data()
        if dataset.data_dict.get("joints2d"):
            joints2d = dataset.data_dict["joints2d"][0]
            vis.set_keypoints_seq([joints2d[t] for t in range(T)])

    render_views = kwargs.get('render_views', ['src_cam', 'above', 'side'])
    src_cam_views = [v for v in render_views if v == 'src_cam']
    other_views = [v for v in render_views if v != 'src_cam']

    for res_dict, out_name in zip(res_dicts, out_names):
        res_dict = move_to(res_dict, device)

        if src_cam_views:
            scene = prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                hand_model,
                temporal_smooth=cfg.temporal_smooth,
                smooth_trans=False,
            )
            if kwargs.get('render_keypoints', False):
                _set_pred_keypoints(vis, hand_model, res_dict)
            animate_scene(
                vis, scene, out_name, seq_name=dataset.seq_name,
                **{**kwargs, 'render_views': src_cam_views}
            )

        if other_views:
            scene = prep_result_vis(
                res_dict,
                obs_data["vis_mask"],
                obs_data["track_id"],
                hand_model,
                temporal_smooth=cfg.temporal_smooth,
                smooth_trans=True,
            )
            animate_scene(
                vis, scene, out_name, seq_name=dataset.seq_name,
                **{**kwargs, 'render_views': other_views, 'render_keypoints': False}
            )

    vis.close()



