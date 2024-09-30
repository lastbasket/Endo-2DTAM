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
from utils.depth_to_normal import depth_to_normal

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
    print(path)
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

def vis_dep(dep, scale=655.35):
    dep = dep.astype(np.float64)
    min_dep = 10
    max_val = 1/min_dep
    alpha = 255/max_val
    dep[dep>0] = scale/dep[dep>0]
    dep_color = cv2.convertScaleAbs(dep, alpha=alpha)
    dep_color = cv2.applyColorMap(dep_color, cv2.COLORMAP_JET)
    return dep_color

def vis_normal(normal):
    normal_save = normal
    normal_save[..., 1] = -normal_save[..., 1]
    normal_save[..., 2] = -normal_save[..., 2]
    normal_save = ((normal_save + 1)/2*255).astype(np.uint8)
    return normal_save

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
    print(args.data_config)
    data_config = load_dataset_config(args.data_config)
    dataset = data_config["dataset_name"].upper()
    
    scene_dir = os.path.join(args.render, scene_name)
    color_render = os.path.join(scene_dir, 'eval', 'color')
    depth_render = os.path.join(scene_dir, 'eval', 'depth')
    normal_render = os.path.join(scene_dir, 'eval', 'normal')
    recon_render = os.path.join(scene_dir, 'eval', 'vis_recon')
    recon_render_gray = os.path.join(scene_dir, 'eval', 'vis_recon_gray')
    

    scene_name = scene_dir.split('/')[-1]
    
    color_render_gt = os.path.join('data', dataset, scene_name, 'color')
    depth_render_gt = os.path.join('data', dataset, scene_name, 'depth')
    recon_render_gt = os.path.join(scene_dir, 'eval', 'vis_recon_gt')
    recon_render_gt_gray = os.path.join(scene_dir, 'eval', 'vis_recon_gt_gray_gt')
    
    color_files2 = lsFile(color_render)
    depth_files2 = lsFile(depth_render, 'tiff')
    recon_files2 = lsFile(recon_render)
    recon_gray_files2 = lsFile(recon_render_gray)
    normal_files2 = lsFile(normal_render)
    
    color_files2_gt = lsFile(color_render_gt)
    depth_files2_gt = lsFile(depth_render_gt, 'tiff')
    recon_files2_gt = lsFile(recon_render_gt)
    recon_files2_gt_gray = lsFile(recon_render_gt_gray)
    
    color2 = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in color_files2]
    normal = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in normal_files2]
    recon = [cv2.imread(image_path, cv2.IMREAD_COLOR)[(800-540)//2:(800-(800-540)//2),
                                                      (800-675)//2:(800-(800-675)//2)-1, :] for image_path in recon_files2]
    recon_gray = [cv2.imread(image_path, cv2.IMREAD_COLOR)[(800-540)//2:(800-(800-540)//2),
                                                      (800-675)//2:(800-(800-675)//2)-1, :] for image_path in recon_gray_files2]
    depth = [cv2.imread(image_path, -1) for image_path in depth_files2]
    
    color2_gt = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in color_files2_gt]
    recon_gt = [cv2.imread(image_path, cv2.IMREAD_COLOR)[(800-540)//2:(800-(800-540)//2),
                                                      (800-675)//2:(800-(800-675)//2)-1, :] for image_path in recon_files2_gt]
    recon_gt_gray = [cv2.imread(image_path, cv2.IMREAD_COLOR)[(800-540)//2:(800-(800-540)//2),
                                                      (800-675)//2:(800-(800-675)//2)-1, :] for image_path in recon_files2_gt_gray]
    depth_gt = [cv2.imread(image_path, -1) for image_path in depth_files2_gt]
    normal_gt = [vis_normal(depth_to_normal(torch.tensor(dep).float().cuda()[None])[0].detach().cpu().numpy()) for dep in depth_gt]
    
    all_idx = set(range(len(color2_gt)))
    train_idx = None
    eval_idx = set(range(7, len(color2_gt), 8)) # we don't want to expect eval in early frames
    train_idx = all_idx - eval_idx
    eval_idx = sorted(list(eval_idx))
    train_idx = sorted(list(train_idx))
    color2_gt = np.array(color2_gt)[eval_idx]
    depthmaps_gt = np.array([vis_dep(image, scale=2.55) for image in depth_gt])[eval_idx]
    normal_gt = np.array(normal_gt)[eval_idx]
    recon_gt = np.array(recon_gt)
    recon_gt_gray = np.array(recon_gt_gray)
    
    rgbmaps = [image for image in color2]
    normalmaps = [image for image in normal]
    reconmaps = [image for image in recon]
    depthmaps = [vis_dep(image) for image in depth]
    
    frame_time = 174+170
    fps = 10000/frame_time
    frameSize = (675, 540)
    out_rgb = cv2.VideoWriter(os.path.join(scene_dir, 'output_rgb.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_dep = cv2.VideoWriter(os.path.join(scene_dir, 'output_dep.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_normal = cv2.VideoWriter(os.path.join(scene_dir, 'output_normal.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_recon = cv2.VideoWriter(os.path.join(scene_dir, 'output_recon.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_recon_gray = cv2.VideoWriter(os.path.join(scene_dir, 'output_recon_gray.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    
    out_rgb_gt = cv2.VideoWriter(os.path.join(scene_dir, 'output_rgb_gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_dep_gt = cv2.VideoWriter(os.path.join(scene_dir, 'output_dep_gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_normal_gt = cv2.VideoWriter(os.path.join(scene_dir, 'output_normal_gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_recon_gt = cv2.VideoWriter(os.path.join(scene_dir, 'output_recon_gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    out_recon_gray_gt = cv2.VideoWriter(os.path.join(scene_dir, 'output_recon_gray_gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    

    # Read images from a folder
    for i in range(len(rgbmaps)):
        out_rgb.write(rgbmaps[i])
        out_dep.write(depthmaps[i])
        out_normal.write(normalmaps[i])
        out_recon.write(reconmaps[i])
        out_recon_gray.write(recon_gray[i])
        
        out_rgb_gt.write(color2_gt[i])
        out_dep_gt.write(depthmaps_gt[i])
        out_normal_gt.write(normal_gt[i])
        out_recon_gt.write(recon_gt[i])
        out_recon_gray_gt.write(recon_gt_gray[i])

    # Release the VideoWriter object
    out_rgb.release()
    out_dep.release()
    out_normal.release()
    out_recon.release()
    out_recon_gray.release()
    
    out_rgb_gt.release()
    out_dep_gt.release()
    out_normal_gt.release()
    out_recon_gt.release()
    out_recon_gray_gt.release()
    