import copy
from copy import deepcopy

model_and_render = dict(
    # diffusion='sd-1.5',  # sd-1.5, sd-2.1, imagen
    diffusion='multiview',  # sd-1.5, sd-2.1, imagen, multiview

    num_voxels=160**3,           # expected number of voxel
    num_background_voxels=4096,
    num_voxels_base=160**3,      # to rescale delta distance
    fast_color_thres=1e-2,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=False,    # maskout grid points that between cameras and their near planes
    stepsize=.5,
    latent=False,
    latent_psuedo=True,
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    viewbase_pe=4,
    model_type='dvgo',  # dvgo dvgo_imp
    avatar_type='full',  # full half shoulder head

)
diffusion_config = dict(
    imagen=dict(
        height=64,
        width=64
    ),
    sd=dict(
        height=512,
        width=512
    ),
    multiview=dict(
        height=256,
        width=256
    )
)

if model_and_render['latent']:
    diffusion_config['multiview']['height'] = diffusion_config['multiview']['height'] // 8
    diffusion_config['multiview']['width'] = diffusion_config['multiview']['width'] // 8

diffusion_type = model_and_render['diffusion'].split('-')[0]
# expname = 'ablation_link02-no_focus'
# expname = 'reference_only-full-lalaland_Wilde-2-scale_0.9-textureless'
expname = 'voxel120_rgb6_width64_bgvoxel2048_resolution384x384'
# expname = 'link-iter5000-smooth2000-albedo1000-full'
# expname='guidance_100-batch_32-thresh_0.1-tighten_bbox_twice-lr_decay-albedo2000-fix_normal_init2-pg_scale6'

# expname='Kratos-iter5000-smooth2000-albedo1000-%s' % coarse_model_and_render['avatar_type']

# expname = 'test-view'
basedir = './logs/%s/dense_pose-3d/' % diffusion_type

prompt = 'cute boy,indifference,white_hair,sweatshirt,Cowboy hat'
data = dict(
    datadir='',
    dataset_type='diffusion-multi-diffusion',
    white_bkgd=False,
    n_in_one=4,
    # text='A photo of a face_token with green hair, angry face, wearing light blue T-shirt',  # face_token
    text='Elsa in Frozen Disney',  # face_token
    negative_text='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
    
    # negative_text='',
    dir_text=False,
    pose_size=1,
    focal_range=[0.7, 1.35],
    pose_type='nerf',
    max_batch_size=1,
    height=diffusion_config[diffusion_type]['height'],
    width=diffusion_config[diffusion_type]['width'],
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)

)

train = dict(
    time=False,
    render_2d=False,
    N_iters=100//1,                 # number of optimization steps
    N_rand=64*64*8*8*4,                  # batch size (number of random rays per optimization step)
    N_img=4,                   # batch size (number of random images per optimization step)
    guidance=10,  # guidance scale
    # max_guidance=100.,

    warm_up_guidance=30,
    warm_up_guidance_step=0,
    lrate_density=0,           # lr of density voxel grid
    lrate_k0=0.1*1,                # lr of color/feature voxel grid
    lrate_background=0.005*1,      # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,               # not used

    warmup_lr_rate=0.0001,
    warmup_step=0,
    exp_decay_start_step=1000//1,
    lrate_decay=5000//1,               # lr decay by 0.1 after every lrate_decay*1000 steps

    ray_sampler='diffusion',
    
)
