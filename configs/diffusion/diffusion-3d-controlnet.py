import copy
from copy import deepcopy

coarse_model_and_render = dict(
    diffusion='controlnet', 
    diffusion_path = 'controlnet/configs/control_dense_pose.ckpt',
    num_voxels=120**3,           # expected number of voxel
    num_voxels_base=120**3,      # to rescale delta distance
    fast_color_thres=1e-2,        # threshold of alpha value to skip the fine stage sampled point
    stepsize=.5,
    latent=False,
    rgbnet_dim=6,                 # feature voxel grid dim
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=64,             # width of the colors MLP
    model_type='dvgo',  # dvgo dvgo_imp
    avatar_type='full',  # full shoulder head

)
diffusion_config = dict(
    sd=dict(
        height=384,
        width=384
    ),
    controlnet=dict(
        height=384,
        width=384
    )
)

diffusion_type = coarse_model_and_render['diffusion'].split('-')[0]

expname = 'test'
basedir = './logs/%s/dense_pose-3d/' % diffusion_type

data = dict(
    dataset_type='diffusion-avatar',
    white_bkgd=False,
    text='Elsa in Frozen Disney',  
    # text='Link in Zelda',  
    # text='Captain America',  
    # text='An iron man',  
    # text='A spider man',  
    # text='Buzz Lightyear from Toy Story',  
    # more prompts can be found in our project page https://avatarverse3d.github.io/ (right-bottom corner of each video in gallery)
    negative_text='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',

    dir_text=True,
    pose_size=100000,
    focal_range=[0.7, 1.35],
    pose_type='nerf',
    max_batch_size=32,
    height=diffusion_config[diffusion_type]['height'],
    width=diffusion_config[diffusion_type]['width'],
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    radius_scale=1.5 if coarse_model_and_render['avatar_type'] == 'head' else 1.0,

)

coarse_train = dict(
    time=False,
    N_iters=4000,                  # number of optimization steps
    N_rand=64*64*8*8,              # batch size (number of random rays per optimization step)
    N_img=3,                       # batch size (number of random images per optimization step)
    guidance=100,                  # guidance scale

    lrate_density=0.1*1,           # lr of density voxel grid
    lrate_k0=0.1*1,                # lr of color/feature voxel grid
    lrate_background=0.005*1,      # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,             # lr of rgbnet


    exp_decay_start_step=1000,
    lrate_decay=5000,               # lr decay by 0.1 after every lrate_decay steps

    ray_sampler='diffusion',
    weight_loss_smooth=10. if coarse_model_and_render['avatar_type'] in ['head', 'shoulder'] else 1.,


    pg_scale=[500, 1500, 2000],
    # pg_scale=[],
    pg_scale_factor=2,
    tighten_bbox=[3000],
    # tighten_bbox=[],
    tighten_thresh=[0.1],
    radius_stage=[1000, 2000],  # 200, 400
    # radius_stage=[],
    tmax=0.98,  # 0.48， 0.98
    focus_start_iter=1000,  #

    mask_step=2000,
    albedo_iters=1000,
    smooth_iters=2000,
)

fine_train = dict(
    N_iters=2000,                 # number of optimization steps
    N_img=4,                   # batch size (number of random images per optimization step)
    guidance=100,  # guidance scale
    lrate_defsdf=0.001*1,           # lr of sdf and deformation
    lrate_k0=0.1*0.1,                # lr of color/feature voxel grid
    lrate_background=1e-3,                # lr of color/feature voxel grid
    lrate_rgbnet=0.,

    exp_decay_start_step=10000,
    lrate_decay=10000,               # lr decay by 0.1 after every lrate_decay steps

    ray_sampler='diffusion',
    weight_loss_smooth=1.*10,
    
    tmax=0.48,  # 0.48， 0.98
    
    pg_scale=[],
    pg_scale_factor=2,
    radius_stage=[500, 1000],  # 500, 1000
    focus_start_iter=0,  #

    mask_step=10000,
    albedo_iters=0,
    smooth_iters=0,
)

fine_model_and_render = copy.deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    # diffusion='sd-1.5',  # sd-1.5, sd-2.1, imagen
    diffusion='controlnet',  # sd-1.5, sd-2.1, controlnet

    num_voxels=100,  # grid res
    latent=False,
    model_type='dmtet',  # dvgo dvgo_imp
    coarse_model_type=coarse_model_and_render['model_type'],  # dvgo dvgo_imp
    stepsize=0.5,
)
)
diffusion_type_fine = fine_model_and_render['diffusion'].split('-')[0]
data_fine = deepcopy(data)
data_fine['height'] = diffusion_config[diffusion_type_fine]['height']
data_fine['width'] = diffusion_config[diffusion_type_fine]['width']
# data_fine['pose_size'] = 2
data_fine['pose_type'] = 'tetra'
data_fine['dataset_type'] = 'diffusion-avatar-full'
data_fine['focal_range'] = [0.7, 1.35]  # [0.7, 1.35]， [1.2, 1.8]
