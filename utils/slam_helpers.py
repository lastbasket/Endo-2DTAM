import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation, remove_points
from diff_surfel_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera, energy_mask
from utils.point_utils import get_pointcloud
import numpy as np
from datasets.gradslam_datasets import (
    EndoSLAMDataset,
    C3VDDataset,
    SCAREDDataset,
    SimCol3DDataset
)

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# def params2rendervar(params):
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': params['rgb_colors'],
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    if params['log_scales'].shape[1] == 1:
        rendervar['colors_precomp'] = params['rgb_colors']
        rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
    else:
        rendervar['shs'] = torch.cat(
            (params['rgb_colors'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2), 
             params['feature_rest'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2)), 
            dim=1)
        rendervar['scales'] = torch.exp(params['log_scales'])
    return rendervar


# def project_points(points_3d, intrinsics):
#     """
#     Function to project 3D points to image plane.
#     params:
#     points_3d: [num_gaussians, 3]
#     intrinsics: [3, 3]
#     out: [num_gaussians, 2]
#     """
#     points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
#     points_2d = points_2d.transpose(0, 1)
#     points_2d = points_2d / points_2d[:, 2:]
#     points_2d = points_2d[:, :2]
#     return points_2d

# def params2silhouette(params):
#     sil_color = torch.zeros_like(params['rgb_colors'])
#     sil_color[:, 0] = 1.0
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': sil_color,
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


# def transformed_params2silhouette(params, transformed_pts):
#     sil_color = torch.zeros_like(params['rgb_colors'])
#     sil_color[:, 0] = 1.0
#     rendervar = {
#         'means3D': transformed_pts,
#         'colors_precomp': sil_color,
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette


# def params2depthplussilhouette(params, w2c):
#     rendervar = {
#         'means3D': params['means3D'],
#         'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
#         'rotations': F.normalize(params['unnorm_rotations']),
#         'opacities': torch.sigmoid(params['logit_opacities']),
#         'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
#         'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
#     }
#     return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    if params['log_scales'].shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
    else:
        rendervar['scales'] = torch.exp(params['log_scales'])
    return rendervar


def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts, rel_w2c


def transform_to_frame_eval(params, camrt=None, rel_w2c=None):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if rel_w2c is None:
        cam_rot, cam_tran = camrt
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts


# def mask_params(params, mask):
#     params['means3D'] = params['means3D'][mask]
#     params['rgb_colors'] = params['rgb_colors'][mask]
#     params['unnorm_rotations'] = params['unnorm_rotations'][mask]
#     params['logit_opacities'] = params['logit_opacities'][mask]
#     params['log_scales'] = params['log_scales'][mask]    
#     return params

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True, config=None):
    use_dep = config['use_dep']
    # Silhouette Rendering
    transformed_pts, _ = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_pts)
    im, _, allmap = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    silhouette = allmap[1:2]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    if use_dep:
        gt_depth = curr_data['depth'][0, :, :]
        render_depth = allmap[0:1]
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        # render depth is too far
        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
        # Determine non-presence mask
        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    else:
        non_presence_mask = non_presence_sil_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        if use_dep:
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        else:
            non_presence_mask = non_presence_mask
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    return params, variables

# def remove_floating_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True, config=None):
#     use_dep = config['use_dep']
#     # Silhouette Rendering
#     transformed_pts, _ = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
#     rendervar = transformed_params2rendervar(params, curr_data['w2c'],
#                                                                  transformed_pts)
    
    
#     im, _, allmap = Renderer(raster_settings=curr_data['cam'])(**rendervar)
#     silhouette = allmap[1:2]
#     non_presence_opa_mask = (silhouette < sil_thres)
    
#     # Check for new foreground objects by using GT depth
#     if use_dep:
#         gt_depth = curr_data['depth'][0, :, :]
#         render_depth = allmap[0:1]
#         depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
#         # render depth is too closed
#         floating_depth_mask = (render_depth < gt_depth) * (depth_error < 20*depth_error.mean())
#         # Determine non-presence mask
#         non_presence_mask = non_presence_opa_mask | floating_depth_mask
#     else:
#         non_presence_mask = non_presence_opa_mask
        
#     # shape is wrong 
#     # Flatten mask
#     non_presence_mask = non_presence_mask.reshape(-1)
#     to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()

#     # Get the new frame Gaussians based on the Silhouette
#     if torch.sum(non_presence_mask) > 0:
#         # Get the new pointcloud in the world frame
#         curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
#         curr_cam_tran = params['cam_trans'][..., time_idx].detach()
#         curr_w2c = torch.eye(4).cuda().float()
#         curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
#         curr_w2c[:3, 3] = curr_cam_tran
#         if use_dep:
#             valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
#             non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
#         else:
#             non_presence_mask = non_presence_mask
#         valid_color_mask = energy_mask(curr_data['im']).squeeze()
#         non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
#         new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
#                                     curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
#                                     mean_sq_dist_method=mean_sq_dist_method)
#         new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
#         for k, v in new_params.items():
#             params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
#         num_pts = params['means3D'].shape[0]
#         variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
#         variables['denom'] = torch.zeros(num_pts, device="cuda").float()
#         variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
#         new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
#         variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
#     return params, variables



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scared"]:
        return SCAREDDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["simcol3d"]:
        return SimCol3DDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
    
def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=False):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 2)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, exp=None):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k != 'feature_rest']
    if 'feature_rest' in params:
        param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})
    if exp is not None:
        param_groups += [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in exp.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def update_optimizer(exp, lrs, optimizer):
    exist = False
    for k, v in exp.items():
        for group in optimizer.param_groups:
            if group["name"] == k:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(v)
                stored_state["exp_avg_sq"] = torch.zeros_like(v)
                exist=True        
                del optimizer.state[group['params'][0]]
                group["params"][0] = v
                optimizer.state[group['params'][0]] = stored_state
    if not exist:
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in exp.items()]
        for group in param_groups:
            optimizer.add_param_group(group)
                
    return optimizer

def prepare_dam(img, norm_mean, norm_std):
        return (img-norm_mean[:, None, None]) / norm_std[:, None, None]
    
def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, use_simplification=True, config=None):
    use_dep = config['use_dep']
    if not use_dep:
        use_dam = config['use_dam']
            
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)

    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio # NOTE: change_here

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam
    

def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 2)),
    }
    if not use_simplification:
        # [n, 3, 15]
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            
            # prev_1 = torch.eye(4, dtype=torch.float32)[None]
            # prev_2 = torch.eye(4, dtype=torch.float32)[None]
            # prev_1[:, :3, :3] = build_rotation(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            # prev_2[:, :3, :3] = build_rotation(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            
            # prev_1[:, :3, 3] = params['cam_trans'][..., curr_time_idx-1].detach()
            # prev_2[:, :3, 3] = params['cam_trans'][..., curr_time_idx-2].detach()
            
            # new_SE3 = prev_1 @ prev_2.inverse() @ prev_1
            
            # new_rot = matrix_to_quaternion(new_SE3[:, :3, :3])
            # new_tran = new_SE3[:, :3, 3]
            
            
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            
            
            # params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            # params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params