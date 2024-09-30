import os

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
    output_name = str(os.environ["OUTPUT_NAME"])
except KeyError:
    output_name = "C3VD_base"
    
try:    
    down_scale = int(os.environ["DOWN_SCALE"])
except KeyError:
    down_scale = 2

map_every = 1 if down_scale == 2 else 2
keyframe_every = 8 if down_scale == 2 else 4
# mapping_window_size = 24
tracking_iters = 15 if down_scale == 2 else 8
mapping_iters = 15 if down_scale == 2 else 8
do_ba=False
ba_iters = 200
ba_every = 100

group_name = output_name
run_name = scene_name

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    ba_every=ba_every,
    keyframe_every=keyframe_every, # Keyframe every nth frame
    distance_keyframe_selection=True, # Use Naive Keyframe Selection
    distance_current_frame_prob=0.1 if down_scale == 2  else 0.5, #0.95, # Probability of choosing the current frame in mapping optimization
    mapping_window_size=-1, # Mapping window size
    report_global_progress_every=2000, # Report Global Progress every nth frame
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=int(1e10), # Checkpoint Interval
    data=dict(
        basedir="./data/C3VD",
        gradslam_data_cfg="./configs/data/c3vd.yaml",
        sequence=scene_name,
        desired_image_height=1080//down_scale,
        desired_image_width=1350//down_scale,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        train_or_test="train",
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            point2plane=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.005,
            exp_a=0.01,
            exp_b=0.01
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, #if down_scale == 2 else 0.3, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=1.0,
            depth=1.0,
            normal=1.0,
            depth_dist=1000.0,
        ),
        
        lambda_dist = 0.0,
        lambda_normal = 0.05,
        opacity_cull = 0.05,
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.005,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.000,
            cam_trans=0.000,
            exp_a=0.001,
            exp_b=0.001
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=0,
            stop_after=20,
            prune_every=100,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10), # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    ba=dict(
        do_ba=do_ba,
        num_iters=ba_iters,
        prune_gaussians=True,
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.005,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.002,
            cam_trans=0.005,
            exp_a=0.001,
            exp_b=0.001
        ),
        loss_weights=dict(
            im=1.0,
            depth=1.0,
            normal=1.0,
            depth_dist=1000.0,
            point2plane=1.0,
        ),
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=50,
            prune_every=50,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10), # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=False, # Visualize Camera Frustums and Trajectory
        viz_w=320, viz_h=320,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=30, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        gaussian_simplification=False,
    ),
    use_dep=True,
    use_normal=True,
    
    use_dam=False,
    dam_dep_scale=100
    
)