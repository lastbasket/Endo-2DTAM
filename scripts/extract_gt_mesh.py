import open3d as o3d
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
from utils.metrics_helper import lsFile
import cv2
from PIL import Image
import yaml



def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def read_pose_file(pose_file):
    """
    Reads a pose file and extracts camera poses. Each pose is expected to be in a comma-separated format, representing a 4x4 transformation matrix.

    Args:
        pose_file: The file path to the pose file. The file should contain lines, each line representing a camera pose as 16 comma-separated floats that can be reshaped into a 4x4 matrix.
        Here we use extrinsic to represent the camera pose

    Returns:
        poses: A list of 4x4 numpy arrays, each array representing a camera pose as extracted from the file.
    """
    
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        poses = [np.linalg.inv(np.array([float(x) for x in line.split(',')]).reshape(4, 4)) for line in lines]
        if poses[-1][:3, 3].sum() == 0:
            poses = [pose.T for pose in poses]
            # for i in range(len(poses)):
            #     poses[i][:2, 3] *= -1 # this is for the niceslam coord
            
    
    all_idx = set(range(len(poses)))
    train_idx = None
    eval_idx = set(range(7, len(poses), 8)) # we don't want to expect eval in early frames
    train_idx = all_idx - eval_idx
    eval_idx = sorted(list(eval_idx))
    train_idx = sorted(list(train_idx))
    poses = [poses[i] for i in eval_idx]
    return poses

def to_cam_open3d(extrinsics, intrins, H, W):
    camera_traj = []
    for i, extrinsic in enumerate(extrinsics):
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=W,
            height=H,
            cx = intrins[0,2],
            cy = intrins[1,2], 
            fx = intrins[0,0], 
            fy = intrins[1,1]
        )

        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def get_cam_K(fx, fy, cx, cy):
    """
    Return camera intrinsics matrix K

    Returns:
        K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
    """
    K = as_intrinsics_matrix([fx, fy, cx, cy])
    K = torch.from_numpy(K)
    return K

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def extract_mesh_bounded(scene_dir, data_config, voxel_size=0.1, sdf_trunc=0.5, depth_trunc=100, depth_scale=1, down_scale = 2,  mask_backgrond=True):
    """
    Perform TSDF fusion given a fixed depth range, used in the paper.
    
    voxel_size: the voxel size of the volume
    sdf_trunc: truncation value
    depth_trunc: maximum depth range, should depended on the scene's scales
    mask_backgrond: whether to mask backgroud, only works when the dataset have masks

    return o3d.mesh
    """
    # Do voxel_size = 0.001 sdf_trunc = 0.005, depth_trunc=1, depth_scale=1000
    print("Running tsdf volume integration ...")
    print(f'voxel_size: {voxel_size}')
    print(f'sdf_trunc: {sdf_trunc}')
    print(f'depth_truc: {depth_trunc}')
    
    orig_height = data_config["camera_params"]["image_height"]
    orig_width = data_config["camera_params"]["image_width"]
    desired_height=orig_height//(down_scale//2)
    desired_width=orig_width//(down_scale//2)
    fx = data_config["camera_params"]["fx"]//(down_scale//2)
    fy = data_config["camera_params"]["fy"]//(down_scale//2)
    cx = data_config["camera_params"]["cx"]//(down_scale//2)
    cy = data_config["camera_params"]["cy"]//(down_scale//2)
    dataset = data_config["dataset_name"].upper()
    
    intrinsic = get_cam_K(fx, fy, cx, cy)

    color_render = os.path.join(scene_dir, 'eval', 'color')
    
    depth_render = os.path.join(scene_dir, 'eval', 'depth')
    depth_files2 = lsFile(depth_render, 'tiff')
    scene_name = scene_dir.split('/')[-1]
    pose_file = os.path.join('data', dataset, scene_name, 'pose.txt')
    
    extrinsics = read_pose_file(pose_file)
    
    color_files2 = lsFile(color_render)
    color2 = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in color_files2]
    rgbmaps = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in color2]
    depthmaps = [np.array(Image.open(image_path)).astype(np.float32) / 655.35 for image_path in depth_files2]
    
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length= voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, cam_o3d in tqdm(enumerate(to_cam_open3d(extrinsics, intrinsic, desired_height, desired_width)), desc="TSDF integration progress"):
        rgb = rgbmaps[i]
        depth = depthmaps[i]
        
        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
            depth_scale = depth_scale
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    return mesh
    
    
if __name__ == '__main__':
    scenes = [
        "cecum_t1_b", 
        "cecum_t2_b", 
        "cecum_t3_a", 
        "sigmoid_t1_a", 
        "sigmoid_t2_a", 
        "sigmoid_t3_a", 
        "trans_t1_b", 
        "trans_t2_c", 
        "trans_t4_a", 
        "trans_t4_b"
    ]

    primary_device="cuda:0"
    seed = 0
    try:    
        scene_name = scenes[int(os.environ["SCENE_NUM"])]
    except KeyError:
        scene_name = "cecum_t1_b"
    try:    
        down_scale = int(os.environ["DOWN_SCALE"])
    except KeyError:
        down_scale = 2
        
        
    print('use', scene_name)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="The base folder containing 'color' and 'depth' subfolders.")
    parser.add_argument("--data_config", help="The base folder containing 'color' and 'depth' subfolders.")
    parser.add_argument("--render", help="The comparison folder containing 'color' and 'depth' subfolders.")
    parser.add_argument("--test_single", help="Test single metric function.", action='store_true')
    parser.add_argument("--name", help="Name of record folder", default='')
    args = parser.parse_args()
    data_config = load_dataset_config(args.data_config)
    
    
    mesh = extract_mesh_bounded(os.path.join(args.render, scene_name), data_config, down_scale=down_scale)
    name = 'fuse.ply'
    
    o3d.io.write_triangle_mesh(os.path.join(args.render, scene_name, name), mesh)
    print("mesh saved at {}".format(os.path.join(args.render, scene_name, name)))