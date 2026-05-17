import math
import numpy as np
import cv2
import torch
import torch.nn as nn

from geometry.rotation import rotation_matrix_to_angle_axis
from geometry import camera as cam_util
from optim.bio_loss import BMCLoss
from util.logger import Logger
from typing import List, Tuple, NewType

Tensor = NewType('Tensor', torch.Tensor)

CONTACT_HEIGHT_THRESH = 0.08
openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices

def solid_angles(
    points: Tensor,
    triangles: Tensor,
    thresh: float = 1e-8
) -> Tensor:
    ''' Compute solid angle between the input points and triangles
        Follows the method described in:
        The Solid Angle of a Plane Triangle
        A. VAN OOSTEROM AND J. STRACKEE
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING,
        VOL. BME-30, NO. 2, FEBRUARY 1983
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            solid_angles: BxQxF
                A tensor containing the solid angle between all query points
                and input triangles
    '''
    # Center the triangles on the query points. Size should be BxQxFx3x3
    centered_tris = triangles[:, None] - points[:, :, None, None]

    # BxQxFx3
    norms = torch.norm(centered_tris, dim=-1)

    # Should be BxQxFx3
    cross_prod = torch.cross(
        centered_tris[:, :, :, 1], centered_tris[:, :, :, 2], dim=-1)
    # Should be BxQxF
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (
        norms.prod(dim=-1) +
        dot01 * norms[:, :, :, 2] +
        dot02 * norms[:, :, :, 1] +
        dot12 * norms[:, :, :, 0]
    )
    del dot01, dot12, dot02, norms

    # Should be BxQ
    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle

def winding_numbers(
    points: Tensor,
    triangles: Tensor,
    thresh: float = 1e-8
) -> Tensor:
    ''' Uses winding_numbers to compute inside/outside
        Robust inside-outside segmentation using generalized winding numbers
        Alec Jacobson,
        Ladislav Kavan,
        Olga Sorkine-Hornung
        Fast Winding Numbers for Soups and Clouds SIGGRAPH 2018
        Gavin Barill
        NEIL G. Dickson
        Ryan Schmidt
        David I.W. Levin
        and Alec Jacobson
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            winding_numbers: BxQ
                A tensor containing the Generalized winding numbers
    '''
    # The generalized winding number is the sum of solid angles of the point
    # with respect to all triangles.
    return 1 / (4 * math.pi) * solid_angles(
        points, triangles, thresh=thresh).sum(dim=-1)

def pcl_pcl_pairwise_distance(
    x: torch.Tensor, 
    y: torch.Tensor,
    use_cuda: bool = True,
    squared: bool = False
):
    """
    Calculate the pairse distance between two point clouds.
    """
    
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()

    dtype = torch.cuda.LongTensor if \
        use_cuda else torch.LongTensor

    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz

    if not squared:
        P = torch.clamp(P, min=0.0) # make sure we dont get nans
        P = torch.sqrt(P)
    
    return P

def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img: np.array,
                     keypoints: np.array,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int32)
                keypoint2 = keypoints[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    return img

def render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        #thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1./50 * map_fn(right_hand_keypoints[:, -1])
    else:
        #thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1./50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20]
    pairs = np.array(pairs).reshape(-1,2)

    colors = [100.,  100.,  100.,
              100.,    0.,    0.,
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1,3)
    #colors = np.zeros_like(colors)
    poseScales = [1]
    #img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(img, right_hand_keypoints, pairs, colors, thicknessCircleRatioRight, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    #img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img

def render_body_keypoints(img: np.array,
                          body_keypoints: np.array) -> np.array:
    """
    Render OpenPose body keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """

    thickness_circle_ratio = 1./75. * np.ones(body_keypoints.shape[0])
    thickness_line_ratio_wrt_circle = 0.75
    pairs = []
    pairs = [1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21,11,22,22,23,11,24]
    pairs = np.array(pairs).reshape(-1,2)
    colors = [255.,     0.,     85.,
              255.,     0.,     0.,
              255.,    85.,     0.,
              255.,   170.,     0.,
              255.,   255.,     0.,
              170.,   255.,     0.,
               85.,   255.,     0.,
                0.,   255.,     0.,
              255.,     0.,     0.,
                0.,   255.,    85.,
                0.,   255.,   170.,
                0.,   255.,   255.,
                0.,   170.,   255.,
                0.,    85.,   255.,
                0.,     0.,   255.,
              255.,     0.,   170.,
              170.,     0.,   255.,
              255.,     0.,   255.,
               85.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.]
    colors = np.array(colors).reshape(-1,3)
    pose_scales = [1]
    return render_keypoints(img, body_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)

def render_openpose(img: np.array,
                    hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    #img = render_body_keypoints(img, body_keypoints)
    img = render_hand_keypoints(img, hand_keypoints)
    return img
###################################################

class StageLoss(nn.Module):
    def __init__(self, loss_weights, **kwargs):
        super().__init__()
        self.set_loss_weights(loss_weights)
        self.setup_losses(loss_weights, **kwargs)

    def setup_losses(self, *args, **kwargs):
        raise NotImplementedError

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        Logger.log("Stage loss weights set to:")
        Logger.log(self.loss_weights)


class RootLoss(StageLoss):
    def setup_losses(
        self,
        loss_weights,
        ignore_op_joints=None,
        joints2d_sigma=100,
        use_chamfer=False,
        robust_loss="none",
        robust_tuning_const=4.6851,
        faces=None
    ):
        self.joints2d_loss = Joints2DLoss(ignore_op_joints, joints2d_sigma)
        self.points3d_loss = Points3DLoss(use_chamfer, robust_loss, robust_tuning_const)
        self.inter_penetration_loss = GeneralContactLoss(faces)
        if loss_weights.get("bio", 0.0) > 0.0:
            self.bio_loss = BMCLoss(lambda_bl=1, lambda_rb=1, lambda_a=1)
        else:
            self.bio_loss = None

    def forward(self, observed_data, pred_data, valid_mask=None):
        """
        For fitting just global root trans/orientation.
        Only computes joint/point/vert losses, i.e. no priors.
        """
        stats_dict = dict()
        loss = 0.0

        # Joints in 3D space
        if (
            "joints3d" in observed_data
            and "joints3d" in pred_data
            and self.loss_weights["joints3d"] > 0.0
        ):
            cur_loss = joints3d_loss(
                observed_data["joints3d"], pred_data["joints3d"], valid_mask
            )
            loss += self.loss_weights["joints3d"] * cur_loss
            stats_dict["joints3d"] = cur_loss

        if (
            "joints3d" in pred_data
            and self.loss_weights["bio"] > 0.0
        ):
            cur_loss, _ = self.bio_loss.compute_loss(
                pred_data["joints3d"], valid_mask
            )
            loss += self.loss_weights["bio"] * cur_loss[0]
            stats_dict["bio"] = cur_loss[0]

        # Select vertices in 3D space
        if (
            "verts3d" in observed_data
            and "verts3d" in pred_data
            and self.loss_weights["verts3d"] > 0.0
        ):
            # raise ValueError
            cur_loss = verts3d_loss(
                observed_data["verts3d"], pred_data["verts3d"], valid_mask
            )
            loss += self.loss_weights["verts3d"] * cur_loss
            stats_dict["verts3d"] = cur_loss

        # All vertices to non-corresponding observed points in 3D space
        if (
            "points3d" in observed_data
            and "points3d" in pred_data
            and self.loss_weights["points3d"] > 0.0
        ):
            # raise ValueError
            cur_loss = self.points3d_loss(
                observed_data["points3d"], pred_data["points3d"]
            )
            loss += self.loss_weights["points3d"] * cur_loss
            stats_dict["points3d"] = cur_loss

        # 2D re-projection loss
        if (
            "joints2d" in observed_data
            and "joints3d_op" in pred_data
            and "cameras" in pred_data
            and self.loss_weights["joints2d"] > 0.0
        ):
            joints2d = cam_util.reproject(
                pred_data["joints3d_op"], *pred_data["cameras"]
            )
            cur_loss = self.joints2d_loss(
                observed_data["joints2d"], joints2d, valid_mask
            )

            loss += self.loss_weights["joints2d"] * cur_loss
            stats_dict["joints2d"] = cur_loss

        # smooth 3d joint motion
        if self.loss_weights["joints3d_smooth"] > 0.0:
            cur_loss = joints3d_smooth_loss(pred_data["joints3d"], valid_mask)
            loss += self.loss_weights["joints3d_smooth"] * cur_loss
            stats_dict["joints3d_smooth"] = cur_loss

        # If we're optimizing cameras, camera reprojection loss
        if "bg2d_err" in pred_data and self.loss_weights["bg2d"] > 0.0:
            raise ValueError
            cur_loss = pred_data["bg2d_err"]
            loss += self.loss_weights["bg2d"] * cur_loss
            stats_dict["bg2d_err"] = cur_loss

        # camera smoothness
        if "cam_R" in pred_data and self.loss_weights["cam_R_smooth"] > 0.0:
            raise ValueError
            cam_R = pred_data["cam_R"]  # (T, 3, 3)
            cur_loss = rotation_smoothness_loss(cam_R[1:], cam_R[:-1])
            loss += self.loss_weights["cam_R_smooth"] * cur_loss
            stats_dict["cam_R_smooth"] = cur_loss

        if "cam_t" in pred_data and self.loss_weights["cam_t_smooth"] > 0.0:
            raise ValueError
            cam_t = pred_data["cam_t"]  # (T, 3, 3)
            cur_loss = translation_smoothness_loss(cam_t[1:], cam_t[:-1])
            loss += self.loss_weights["cam_t_smooth"] * cur_loss
            stats_dict["cam_t_smooth"] = cur_loss

        # Depth constraint: keep hand in front of camera
        if (
            "joints3d" in pred_data
            and "cameras" in pred_data
            and self.loss_weights.get("depth_constraint", 0.0) > 0.0
        ):
            # Extract cam_R and cam_t from cameras tuple
            # cameras = (cam_R, cam_t, cam_f, cam_center)
            cam_R, cam_t, cam_f, cam_center = pred_data["cameras"]
            cur_loss = depth_constraint_loss(
                pred_data["joints3d"],
                cam_R,
                cam_t,
                min_depth=0.0,
                max_depth=999
            )
            loss += self.loss_weights["depth_constraint"] * cur_loss
            stats_dict["depth_constraint"] = cur_loss

        return loss, stats_dict


def rotation_smoothness_loss(R1, R2):
    R12 = torch.einsum("...ij,...jk->...ik", R2, R1.transpose(-1, -2))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2)


def translation_smoothness_loss(t1, t2):
    return torch.sum((t2 - t1) ** 2)


def depth_constraint_loss(joints3d, cam_R, cam_t, min_depth=0.0, max_depth=999):
    """
    Penalize joints that are behind the camera or too far away.
    Ensures hand stays within reasonable depth range in camera space.
    
    :param joints3d (B, T, J, 3) or (T, J, 3) joints in world space
    :param cam_R (B, T, 3, 3) or (T, 3, 3) world-to-camera rotation
    :param cam_t (B, T, 3) or (T, 3) world-to-camera translation
    :param min_depth minimum allowed Z in camera space (default 0.1m)
    :param max_depth maximum allowed Z in camera space (default 5.0m)
    """
    # Handle both (B, T, J, 3) and (T, J, 3) shapes
    if joints3d.ndim == 3:
        # (T, J, 3) -> add batch dimension
        joints3d = joints3d.unsqueeze(0)  # (1, T, J, 3)
    if cam_R.ndim == 3:
        # (T, 3, 3) -> add batch dimension
        cam_R = cam_R.unsqueeze(0)  # (1, T, 3, 3)
    if cam_t.ndim == 2:
        # (T, 3) -> add batch dimension
        cam_t = cam_t.unsqueeze(0)  # (1, T, 3)
    
    # Transform joints to camera space
    B, T, J, _ = joints3d.shape
    joints_cam = torch.einsum("btij,btjk->btik", cam_R, joints3d.transpose(-1, -2)).transpose(-1, -2)
    joints_cam = joints_cam + cam_t[..., None, :]  # (B, T, J, 3)
    
    # Get Z coordinate (depth)
    depth = joints_cam[..., 2]  # (B, T, J)
    
    # Penalize negative depth (behind camera) heavily
    behind_camera_loss = torch.sum(torch.relu(-depth) ** 2)
    
    # Penalize depth outside reasonable range
    too_close_loss = torch.sum(torch.relu(min_depth - depth) ** 2)
    too_far_loss = torch.sum(torch.relu(depth - max_depth) ** 2)
    
    return behind_camera_loss * 100.0 + too_close_loss + too_far_loss


def camera_smoothness_loss(R1, t1, R2, t2):
    """
    :param R1, t1 (N, 3, 3), (N, 3)
    :param R2, t2 (N, 3, 3), (N, 3)
    """
    R12, t12 = cam_util.compose_cameras(R2, t2, *cam_util.invert_camera(R1, t1))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2) + torch.sum(t12**2)

"""
Losses are cumulative
SMPLLoss setup is same as RootLoss
"""

class SMPLLoss(RootLoss):
    def forward(self, observed_data, pred_data, nsteps, valid_mask=None):
        """
        For fitting full shape and pose of SMPL.
        nsteps used to scale single-step losses
        """
        loss, stats_dict = super().forward(
            observed_data, pred_data, valid_mask=valid_mask
        )

        if "latent_pose" in pred_data and self.loss_weights["pose_prior"] > 0.0:
            latent_pose_init = observed_data["init_latent_pose"]
            cur_loss = pose_prior_loss(pred_data["latent_pose"], latent_pose_init, valid_mask)
            loss += self.loss_weights["pose_prior"] * cur_loss
            stats_dict["pose_prior"] = cur_loss

        if "betas" in pred_data and self.loss_weights["shape_prior"] > 0.0:
            cur_loss = shape_prior_loss(pred_data["betas"])
            loss += self.loss_weights["shape_prior"] * nsteps * cur_loss
            stats_dict["shape_prior"] = cur_loss

        if 'verts3d' in pred_data and self.loss_weights["penetration"] > 0.0:
            order = torch.sum(pred_data['is_right'], dim=-1)//pred_data['is_right'].shape[1]

            if len(order) > 1:
                cur_loss = 0
                if order[0] == 0:
                    l_verts = pred_data["verts3d"][0]
                    r_verts = pred_data["verts3d"][1]
                else:
                    l_verts = pred_data["verts3d"][1]
                    r_verts = pred_data["verts3d"][0]

                for idx in range(len(l_verts)):
                    cur_loss += self.inter_penetration_loss(v1=l_verts[idx:idx+1], v2=r_verts[idx:idx+1])

                loss += self.loss_weights["penetration"] * cur_loss
                stats_dict["penetration"] = cur_loss

        return loss, stats_dict


def joints3d_loss(joints3d_obs, joints3d_pred, mask=None):
    """
    :param joints3d_obs (B, T, J, 3)
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = joints3d_obs.shape
    vis_mask = get_visible_mask(joints3d_obs)
    if mask is not None:
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (joints3d_obs[vis_mask] - joints3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def verts3d_loss(verts3d_obs, verts3d_pred, mask=None):
    """
    :param verts3d_obs (B, T, V, 3)
    :param verts3d_pred (B, T, V, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = verts3d_obs.shape
    vis_mask = get_visible_mask(verts3d_obs)
    if mask is not None:
        assert mask.shape == (B, T)
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (verts3d_obs[vis_mask] - verts3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def get_visible_mask(obs_data):
    """
    Given observed data gets the mask of visible data (that actually contributes to the loss).
    """
    return torch.logical_not(torch.isinf(obs_data))


class Joints2DLoss(nn.Module):
    def __init__(self, ignore_op_joints=None, joints2d_sigma=100, normalize_by_scale=True):
        super().__init__()
        self.ignore_op_joints = ignore_op_joints
        self.joints2d_sigma = joints2d_sigma
        self.normalize_by_scale = normalize_by_scale

    def forward(self, joints2d_obs, joints2d_pred, mask=None):
        """
        :param joints2d_obs (B, T, 25, 3)
        :param joints2d_pred (B, T, 22, 2)
        :param mask (optional) (B, T)
        """
        # Safety check: detect NaN in predictions before computing loss
        if torch.isnan(joints2d_pred).any() or torch.isinf(joints2d_pred).any():
            return torch.tensor(1e8, device=joints2d_pred.device, requires_grad=True)
        
        if mask is not None:
            mask = mask.bool()
            joints2d_obs = joints2d_obs[mask]  # (N, 25, 3)
            joints2d_pred = joints2d_pred[mask]  # (N, 22, 2)

        joints2d_obs_conf = joints2d_obs[..., 2:3]
        if self.ignore_op_joints is not None:
            # set confidence to 0 so not weighted
            joints2d_obs_conf[..., self.ignore_op_joints, :] = 0.0

        # Compute error
        error = joints2d_pred - joints2d_obs[..., :2]  # (N, 22, 2)
        
        # Normalize by hand scale to make loss resolution-independent
        if self.normalize_by_scale:
            # Estimate hand scale from observed keypoints spread
            valid_obs = joints2d_obs[:, :, :2]  # (N, 25, 2)
            # Compute bbox size of observed keypoints for each hand
            kp_min = valid_obs.min(dim=1, keepdim=True)[0]  # (N, 1, 2)
            kp_max = valid_obs.max(dim=1, keepdim=True)[0]  # (N, 1, 2)
            hand_scale = torch.sqrt(((kp_max - kp_min) ** 2).sum(dim=-1, keepdim=True))  # (N, 1, 1)
            
            # Safety check: detect invalid/degenerate cases
            if torch.isnan(hand_scale).any() or torch.isinf(hand_scale).any() or (hand_scale < 5.0).any():
                return torch.tensor(1e6, device=hand_scale.device, requires_grad=True)
            
            # Clamp to reasonable range (5-1000 pixels)
            hand_scale = torch.clamp(hand_scale, min=5.0, max=1000.0)
            
            # Normalize error by hand scale AND scale up to reasonable magnitude
            error = error / hand_scale * 100.0  # Now error is in normalized units (0-100 scale)
            
            # Use FIXED sigma in normalized space (not divided by hand_scale!)
            normalized_sigma = self.joints2d_sigma  # Already in normalized units
        else:
            normalized_sigma = self.joints2d_sigma

        # weight errors by detection confidence
        robust_sqr_dist = gmof(error, normalized_sigma)
        reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
        loss = torch.mean(reproj_err)
        return loss
    

class Points3DLoss(nn.Module):
    def __init__(
        self,
        use_chamfer=False,
        robust_loss="bisquare",
        robust_tuning_const=4.6851,
    ):
        super().__init__()

        if not use_chamfer:
            self.active = False
            return

        self.active = True

        robust_choices = ["none", "bisquare", "gm"]
        if robust_loss not in robust_choices:
            Logger.log(
                "Not a valid robust loss: %s. Please use %s"
                % (robust_loss, str(robust_choices))
            )
            exit()

        from utils.chamfer_distance import ChamferDistance

        self.chamfer_dist = ChamferDistance()

        self.robust_loss = robust_loss
        self.robust_tuning_const = robust_tuning_const

    def forward(self, points3d_obs, points3d_pred):
        if not self.active:
            return torch.tensor(0.0, dtype=torch.float32, device=points3d_obs.device)

        # one-way chamfer
        B, T, N_obs, _ = points3d_obs.size()
        N_pred = points3d_pred.size(2)
        points3d_obs = points3d_obs.reshape((B * T, -1, 3))
        points3d_pred = points3d_pred.reshape((B * T, -1, 3))

        obs2pred_sqr_dist, pred2obs_sqr_dist = self.chamfer_dist(
            points3d_obs, points3d_pred
        )
        obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((B, T * N_obs))
        pred2obs_sqr_dist = pred2obs_sqr_dist.reshape((B, T * N_pred))

        weighted_obs2pred_sqr_dist, w = apply_robust_weighting(
            obs2pred_sqr_dist.sqrt(),
            robust_loss_type=self.robust_loss,
            robust_tuning_const=self.robust_tuning_const,
        )

        loss = torch.sum(weighted_obs2pred_sqr_dist)
        loss = 0.5 * loss
        return loss


######################
# losses
######################
class GeneralContactLoss(nn.Module):
    def __init__(
        self,
        faces,
        region_aggregation_type: str = 'sum',
        squared_dist: bool = False,
        model_type: str = 'mano',
        body_model_utils_folder: str = 'body_model_utils',
        **kwargs
    ):
        super().__init__()
        """
        Compute intersection and contact between two meshes and resolves.
        """
    
        self.region_aggregation_type = region_aggregation_type
        self.squared = squared_dist

        self.criterion = self.init_loss()       

        # create extra vertex and faces to close back of the mouth to maske
        # the mano mesh watertight.
        self.model_type = model_type

        if self.model_type == 'mano':
            # add faces that make the hand mesh watertight
            faces_new = torch.tensor([[92, 38, 234],
                                [234, 38, 239],
                                [38, 122, 239],
                                [239, 122, 279],
                                [122, 118, 279],
                                [279, 118, 215],
                                [118, 117, 215],
                                [215, 117, 214],
                                [117, 119, 214],
                                [214, 119, 121],
                                [119, 120, 121],
                                [121, 120, 78],
                                [120, 108, 78],
                                [78, 108, 79]])

            r_faces = torch.cat([faces, faces_new.to(faces.device)], dim=0)
            l_faces = r_faces[:,[0,2,1]].clone()

            self.register_buffer('l_faces', l_faces)
            self.register_buffer('r_faces', r_faces)

        # low resolution mesh 
        # inner_mouth_verts_path = f'{body_model_utils_folder}/lowres_{model_type}.pkl'
        # self.low_res_mesh = pickle.load(open(inner_mouth_verts_path, 'rb'))

    # def close_mouth(self, v):
    #     mv = torch.mean(v[:,self.vert_ids_wt,:], 1, keepdim=True)
    #     v = torch.cat((v, mv), 1)
    #     return v

    def to_lowres(self, v, is_right):
        v = v
        if is_right == 0:
            t = v[:,self.l_faces,:]
        elif is_right == 1:
            t = v[:,self.r_faces,:]
        else:
            raise ValueError

        return v, t

    def init_loss(self):
        def loss_func(v1, v2, factor=100, wn_batch=True):
            """
            Compute loss between region r1 on meshes v1 and 
            region r2 on mesh v2.
            """

            nn = 1000

            loss = torch.tensor(0.0, device=v1.device)

            if wn_batch:
                # close mouth for self-intersection test
                v1l, t1l = self.to_lowres(v1, 0)
                v2l, t2l = self.to_lowres(v2, 1)

                # compute intersection between v1 and v2
                interior_v1 = winding_numbers(v1, t2l).ge(0.99)
                interior_v2 = winding_numbers(v2, t1l).ge(0.99)

            batch_losses = []
            for bidx in range(v1.shape[0]):
                if not wn_batch:
                    # close mouth for self-intersection test
                    v1l, t1l = self.to_lowres(v1[[bidx]], nn)
                    v2l, t2l = self.to_lowres(v2[[bidx]], nn)

                    # compute intersection between v1 and v2
                    curr_interior_v1 = winding_numbers(v1[[bidx]], t2l.detach()).ge(0.99)[0]
                    curr_interior_v2 = winding_numbers(v2[[bidx]], t1l.detach()).ge(0.99)[0]
                    crit_v1, crit_v2 = torch.any(curr_interior_v1), torch.any(curr_interior_v2)
                else:
                    curr_interior_v1 = interior_v1[bidx]
                    curr_interior_v2 = interior_v2[bidx]
                    crit_v1, crit_v2 = torch.any(interior_v1[bidx]), torch.any(interior_v2[bidx])

                bloss = torch.tensor(0.0, device=v1.device)
                if crit_v1 and crit_v2:
                    # find vertices that are close to each other between v1 and v2
                    #squared_dist = pcl_pcl_pairwise_distance(
                    #    v1[:,interior_v1[bidx],:], v2[:, interior_v2[bidx], :], squared=self.squared
                    #)
                    squared_dist_v1v2 = pcl_pcl_pairwise_distance(
                        v1[[[bidx]],curr_interior_v1,:], v2[[bidx]], squared=self.squared)
                    squared_dist_v2v1 = pcl_pcl_pairwise_distance(
                        v2[[[bidx]], curr_interior_v2, :], v1[[bidx]], squared=self.squared)
 
                    v1_to_v2 = (squared_dist_v1v2[0].min(1)[0] * factor)**2
                    #v1_to_v2 = 10.0 * (torch.tanh(v1_to_v2 / 10.0)**2)
                    bloss += v1_to_v2.sum()
                    
                    v2_to_v1 = (squared_dist_v2v1[0].min(1)[0] * factor)**2
                    #v2_to_v1 = 10.0 * (torch.tanh(v2_to_v1 / 10.0)**2)
                    bloss += v2_to_v1.sum()

                batch_losses.append(bloss)

            # compute loss
            if len(batch_losses) > 0:
                loss = sum(batch_losses) / len(batch_losses)

            return loss

        return loss_func 

    def forward(self, **args):
        losses = self.criterion(**args)
        return losses

def pose_prior_loss(latent_pose_pred, latent_pose_init=None, mask=None):
    """
    :param latent_pose_pred (B, T, D) - optimized latent pose
    :param latent_pose_init (optional) (B, T, D) - initial HaMeR prediction
    :param mask (optional) (B, T)
    """
    if latent_pose_init is not None:
        # Penalize deviation from original HaMeR prediction
        loss = (latent_pose_pred - latent_pose_init)**2
    else:
        # Fallback: prior is isotropic gaussian so take L2 distance from 0
        loss = latent_pose_pred**2
    
    if mask is not None:
        loss = loss[mask.bool()]
    loss = torch.sum(loss)
    return loss


def shape_prior_loss(betas_pred):
    # prior is isotropic gaussian so take L2 distance from 0
    loss = betas_pred**2
    loss = torch.sum(loss)
    return loss


def joints3d_smooth_loss(joints3d_pred, mask=None, normalize_by_scale=True):
    """
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    :param normalize_by_scale: If True, normalize by hand scale for size-invariance
    """
    # Safety check: detect NaN in predictions before computing loss
    if torch.isnan(joints3d_pred).any() or torch.isinf(joints3d_pred).any():
        return torch.tensor(1e8, device=joints3d_pred.device, requires_grad=True)
    
    B, T, *dims = joints3d_pred.shape
    
    # Normalize by hand scale FIRST to make loss size-invariant
    if normalize_by_scale:
        # Estimate hand scale from 3D joint positions at each frame
        # Use bbox of joints as scale reference
        joints_min = joints3d_pred.min(dim=2, keepdim=True)[0]  # (B, T, 1, 3)
        joints_max = joints3d_pred.max(dim=2, keepdim=True)[0]  # (B, T, 1, 3)
        hand_scale = torch.sqrt(((joints_max - joints_min) ** 2).sum(dim=-1, keepdim=True))  # (B, T, 1, 1)
        
        # Safety check: detect invalid/degenerate cases
        if torch.isnan(hand_scale).any() or torch.isinf(hand_scale).any() or (hand_scale < 0.001).any():
            return torch.tensor(1e6, device=hand_scale.device, requires_grad=True)
        
        # Clamp to reasonable range (0.001m - 1m for hand size)
        hand_scale = torch.clamp(hand_scale, min=0.001, max=1.0)
        
        # Normalize joints FIRST by their own scale at each frame (same as joints2d)
        joints3d_pred_normalized = joints3d_pred / hand_scale  # (B, T, J, 3) - same scale factor as joints2d
    else:
        joints3d_pred_normalized = joints3d_pred
    
    # Now compute delta on normalized joints
    delta = joints3d_pred_normalized[:, 1:, :, :] - joints3d_pred_normalized[:, :-1, :, :]  # (B, T-1, J, 3)
    
    loss = delta ** 2
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]

    loss = torch.mean(loss, dim=0)
    loss = 0.5 * torch.sum(loss)
    return loss




def apply_robust_weighting(
    res, robust_loss_type="bisquare", robust_tuning_const=4.6851
):
    """
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    """
    robust_choices = ["none", "bisquare"]
    if robust_loss_type not in robust_choices:
        print(
            "Not a valid robust loss: %s. Please use %s"
            % (robust_loss_type, str(robust_choices))
        )

    w = None
    detach_res = (
        res.clone().detach()
    )  # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == "none":
        w = torch.ones_like(detach_res)
    elif robust_loss_type == "bisquare":
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w


def robust_std(res):
    """
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    """
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B, 1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std


def bisquare_robust_weights(res, tune_const=4.6851):
    """
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    """
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    outlier_mask = norm_res >= 1.0

    w = (1.0 - norm_res**2) ** 2
    w[outlier_mask] = 0.0

    return w


def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
