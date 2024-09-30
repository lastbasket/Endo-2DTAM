import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
from glob import glob


def depths_to_points(world_view_transform, full_proj_transform, depthmap, gaussians_grad=False, camera_grad=False):
    c2w = (world_view_transform.T).inverse()
    W, H = depthmap.shape[2], depthmap.shape[1]
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ full_proj_transform
    
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(depth, world_view_transform=None, full_proj_transform=None, gaussians_grad=False, camera_grad=False):
    """
        view: view camera
        depth: depthmap 
    """
    
    if world_view_transform is None:
        world_view_transform = torch.tensor([[ 1.0000000e+00, -5.2154064e-08,  5.9604631e-08,  0.0000000e+00],
                    [-6.3329935e-08,  1.0000000e+00,  1.0430811e-07,  0.0000000e+00],
                    [ 2.9802321e-08, -1.4901161e-08,  9.9999988e-01,  0.0000000e+00],
                    [-3.8146973e-06,  3.0994415e-06,  7.6293936e-06,  1.0000000e+00]]).float().cuda()
    
    if full_proj_transform is None:
        full_proj_transform = torch.tensor([[1.18862069e+00, -7.65937855e-08,  5.96105920e-08,  5.96046306e-08],
                                    [-7.63127872e-08,  1.48497224e+00,  1.04318538e-07,  1.04308107e-07],
                                    [-9.94661450e-03,  1.43202925e-02,  1.00009990e+00,  9.99999881e-01],
                                    [-4.61011496e-06,  4.71184012e-06, -9.99336969e-03,  7.62939362e-06]]).float().cuda()
    points = depths_to_points(world_view_transform, full_proj_transform, depth, gaussians_grad, camera_grad).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def vis_dep(dep, scale=655.35):
    dep = dep.astype(np.float64)
    min_dep = 10
    max_val = 1/min_dep
    alpha = 255/max_val
    dep[dep>0] = scale/dep[dep>0]
    dep_color = cv2.convertScaleAbs(dep, alpha=alpha)
    dep_color = cv2.applyColorMap(dep_color, cv2.COLORMAP_JET)
    return dep_color

if __name__ == '__main__':

    file_path = 'figure_4'

    file_list = os.listdir(file_path)
    for i in file_list:
        if i == 'gt':
            scale=2.55
        else:
            scale=655.35
        dep_list = glob(os.path.join(file_path, i, '*.tiff'))
        for dep_path in dep_list:
            vis_path = dep_path.replace('.tiff', '_vis.png')
            normal_path = dep_path.replace('.tiff', '_normal.png')
            
            dep = cv2.imread(dep_path, -1)
            normal, points = depth_to_normal(torch.tensor(dep).float().cuda()[None])
            normal_save = normal.detach().cpu().numpy()

            normal_save[..., 1] = -normal_save[..., 1]
            normal_save[..., 2] = -normal_save[..., 2]
            normal_save = ((normal_save + 1)/2*255).astype(np.uint8)
            dep_color = vis_dep(dep, scale)
            cv2.imwrite(vis_path, dep_color)
            cv2.imwrite(normal_path, normal_save)