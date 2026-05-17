import torch
import torch
import torch.nn as nn

from body_model import MANO_JOINTS, run_mano
from geometry.rotation import (
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
)

from .params import CameraParams

J_HAND = len(MANO_JOINTS) - 1  # no root


class BaseSceneModel(nn.Module):
    """
    Scene model of sequences of human poses.
    All poses are in their own INDEPENDENT camera reference frames.
    A basic class mostly for testing purposes.

    Parameters:
        batch_size:  number of sequences to optimize
        seq_len:     length of the sequences
        body_model:  MANO hand model
        pose_prior:  VPoser model
        fit_gender:  gender of model (optional)
    """

    def __init__(
        self,
        batch_size,
        seq_len,
        body_model,
        pose_prior,
        # fit_gender="neutral",
        use_init=False,
        opt_cams=False,
        opt_scale=True,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.body_model = body_model
        self.hand_mean = self.body_model.hand_mean

        self.pose_prior = pose_prior
        if self.pose_prior is not None:
            self.latent_pose_dim = self.pose_prior.latentD

        self.num_betas = body_model.num_betas

        self.use_init = use_init
        self.opt_scale = opt_scale
        self.opt_cams = opt_cams
        self.params = CameraParams(batch_size)

    def initialize(self, obs_data, cam_data):

        # initialize cameras
        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
        )

        init_betas = torch.mean(obs_data["init_body_shape"], dim=1)

        init_pose = obs_data["init_body_pose"][:, :, :J_HAND, :]
        init_pose_latent = self.pose2latent(init_pose)

        # transform into world frame
        R_w2c, t_w2c = cam_data["cam_R"], cam_data["cam_t"]
        R_c2w = R_w2c.transpose(-1, -2)
        t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)

        init_rot = obs_data["init_root_orient"]  # (B, T, 3)
        init_rot_mat = angle_axis_to_rotation_matrix(init_rot)
        init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w, init_rot_mat)
        init_rot = rotation_matrix_to_angle_axis(init_rot_mat)

        init_trans_cam = obs_data["init_trans"]  # (B, T, 3)
        init_trans = torch.einsum("tij,btj->bti", R_c2w, init_trans_cam) + t_c2w[None]

        self.params.set_param("init_body_pose", init_pose)
        self.params.set_param("latent_pose", init_pose_latent)
        self.params.set_param("betas", init_betas)
        self.params.set_param("trans", init_trans)
        self.params.set_param("root_orient", init_rot)
        self.params.set_param("is_right", is_right, requires_grad=False)

        obs_data["init_latent_pose"] = init_pose_latent.detach()

    def get_optim_result(self, **kwargs):
        res = self.params.get_dict()
        if "latent_pose" in res:
            res["pose_body"] = self.latent2pose(self.params.latent_pose).detach()

        # add the cameras
        res["cam_R"], res["cam_t"], _, _ = self.params.get_cameras()
        res["intrins"] = self.params.intrins
        return {"world": res}

    def latent2pose(self, latent_pose):
        """
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        """
        if self.pose_prior is not None:
            B, T, _ = latent_pose.size()
            d_latent = self.pose_prior.latentD
            latent_pose = latent_pose.reshape((-1, d_latent))
            body_pose = self.pose_prior.decode(latent_pose, output_type="matrot")
            body_pose = rotation_matrix_to_angle_axis(
                body_pose.reshape((B * T * J_HAND, 3, 3))
            ).reshape((B, T, J_HAND * 3))
            return body_pose + self.hand_mean
        else:
            return latent_pose

    def pose2latent(self, body_pose):
        """
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        """
        if self.pose_prior is not None:
            B, T = body_pose.shape[:2]
            body_pose = body_pose.reshape((-1, J_HAND * 3))
            latent_pose_distrib = self.pose_prior.encode(body_pose - self.hand_mean)
            d_latent = self.pose_prior.latentD
            latent_pose = latent_pose_distrib.mean.reshape((B, T, d_latent))
            return latent_pose
        else:
            return body_pose

    def pred_mano(self, trans, root_orient, body_pose, is_right, betas):
        mano_out = run_mano(self.body_model, trans, root_orient, body_pose, is_right, betas=betas)
        joints3d, verts3d = mano_out["joints"], mano_out["vertices"]

        return {
            "is_right": is_right,
            "points3d": verts3d,  # all vertices
            "verts3d": verts3d,  # keypoint vertices
            "joints3d": joints3d,  # smpl joints
            "joints3d_op": joints3d,  # OP joints
            "l_faces": mano_out["l_faces"],  # index array of faces
            "r_faces": mano_out["r_faces"],  # index array of faces
            "body_pose": mano_out["body_pose"]
        }

    def pred_params_mano(self, is_right, reproj=True):
        body_pose = self.latent2pose(self.params.latent_pose)
        pred_data = self.pred_mano(
            self.params.trans, self.params.root_orient, body_pose, is_right, self.params.betas
        )
        return pred_data
