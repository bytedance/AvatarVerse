import imp
import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import utils, dvgo, dvgo_dmtet
from lib.load_data import Diffusion3dDataset

from sd.sd import StableDiffusion
from sd.sd_utils import prepare_text_embeddings, disable_params_grad
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/controlnet")

from controlnet.cldm.model import create_model, load_state_dict
from human.pose_loader import HumanPose, HumanPoseSMPLX
from human.smpl import DensePoseSMPL
import cv2

from accelerate import Accelerator

# from torch_efficient_distloss import flatten_eff_distloss


def hcat(input_t, dim_split, dim_cat):
    return torch.cat(torch.split(input_t, split_size_or_sections=1, dim=dim_split), dim=dim_cat).squeeze(dim_split)


def prepare_model(model, accelerator, cfg_model):
    if cfg_model.model_type != 'dmtet':
        # must run for all processes
        model.density, model.k0, model.background, model.rgbnet = accelerator.prepare(
            model.density, model.k0, model.background, model.rgbnet
        )
    elif cfg_model.model_type == 'dmtet':
        model.defsdf, model.k0, model.background, model.rgbnet = accelerator.prepare(
            model.defsdf, model.k0, model.background, model.rgbnet
        )
    else:
        raise NotImplementedError

    return model

def unwarp_model(model, accelerator, cfg_model):
    if cfg_model.model_type != 'dmtet':
        model.density = accelerator.unwrap_model(model.density)
        model.k0 = accelerator.unwrap_model(model.k0)
        model.background = accelerator.unwrap_model(model.background)
        model.rgbnet = accelerator.unwrap_model(model.rgbnet)
    elif cfg_model.model_type == 'dmtet':
        model.defsdf = accelerator.unwrap_model(model.defsdf)
        model.k0 = accelerator.unwrap_model(model.k0)
        model.background = accelerator.unwrap_model(model.background)
        model.rgbnet = accelerator.unwrap_model(model.rgbnet)
    else:
        raise NotImplementedError
    
    return model

class PsuedoDataset(Dataset):
    def __init__(self, H, W, data_list, num=-1) -> None:
        self.H = H
        self.W = W
        self.data_list = data_list
        self.num = min(len(self.data_list[0]), num) if num > 0 else len(self.data_list[0])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        f = lambda x:  x[idx].clone() if x is not None else None
        return [f(d) for d in self.data_list]

@torch.no_grad()
def render_viewpoints(cfg, cfg_model, model, dataset, ndc, render_kwargs,
                      shading='albedo',savedir=None, dump_images=False, prefix='',
                      render_factor=0, render_video_flipy=False, render_video_rot90=0):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgbs = []
    depths = []
    normals = []
    controls = []

    if len(dataset) > 10:
        range_f = tqdm
    else:
        range_f = lambda x:x

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                            generator=torch.Generator(device=device))
    ambient_ratio = 1. if shading == 'albedo' else 0.1
    render_size = 512*512
    for data in range_f(dataloader):
        # data = dataset[i]
        if len(data) == 4:
            c2w, _, light_pos, K = data
            control = None
        else:
            c2w, _, light_pos, K, control = data
        H, W = dataset.H, dataset.W

        if render_factor != 0:
            H = int(H / render_factor)
            W = int(W / render_factor)
            K[:, :2, :3] /= render_factor

        c2w = c2w.to(device)
        light_d = light_pos.to(device)

        _, rays_o, rays_d, viewdirs, _ = dvgo.get_diffusion_rays(
            rgb_tr=c2w, train_poses=c2w, HW=(H, W), Ks=K,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        keys = ['rgb_marched', 'depth', 'alphainv_last', 'normal_marched']
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        light_d = light_d.reshape([-1, 3])
        with torch.no_grad():
            if cfg_model.model_type != 'dmtet':
                light_d = light_d.reshape([-1, 1, 1, 3]).repeat([1, H, W, 1]).reshape([-1, 3])
                render_result_chunks = [
                    {k: v for k, v in model(ro, rd, vd, ld, shading=shading, ambient_ratio=ambient_ratio, **render_kwargs).items() if k in keys}
                    for ro, rd, vd, ld in zip(rays_o.split(render_size, 0), rays_d.split(render_size, 0),
                                            viewdirs.split(render_size, 0), light_d.split(render_size, 0)
                                            )
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                    for k in render_result_chunks[0].keys()
                }
            else:
                render_result = model(K, c2w, viewdirs, H, W, light_d, shading, ambient_ratio, **render_kwargs)
                render_result = {
                    k: render_result[k].squeeze(0)
                    for k in render_result.keys()
                }

        rgb = torch.reshape(render_result['rgb_marched'], [1, 1, H, W, -1])
        rgb = hcat(rgb, dim_split=1, dim_cat=3).cpu().numpy().squeeze(0)
        depth = torch.reshape(render_result['depth'], [1, 1, H, W, -1])
        depth = hcat(depth, dim_split=1, dim_cat=3).cpu().numpy().squeeze(0)
        if 'normal_marched' in render_result:
            normal = torch.reshape(render_result['normal_marched'], [1, 1, H, W, -1])
            normal = hcat(normal, dim_split=1, dim_cat=3).cpu().numpy().squeeze(0)
            # normal = render_result['normal_marched'].cpu().numpy()
            normals.append(normal)

        rgbs.append(rgb)
        depths.append(depth)
        if control is not None:
            control = F.interpolate(control, (H, W), mode='bilinear', align_corners=False)
            control = hcat(control, dim_split=0, dim_cat=3)
            controls.append(control.cpu().permute(1, 2, 0).numpy())

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            normals[i] = np.flip(normals[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            normals[i] = np.rot90(normals[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        all_img8 = []
        for i in range(len(rgbs)):
            global_i = i + len(rgbs)
            filename = os.path.join(savedir, (prefix + 'vis_{:03d}.jpg').format(global_i))

            rgb8 = cv2.cvtColor(utils.to8b(rgbs[i]), cv2.COLOR_RGB2BGR)
            normal8 = cv2.cvtColor(utils.to8b(normals[i]/2.+0.5), cv2.COLOR_RGB2BGR)
            depth8 = cv2.cvtColor(utils.to8b(1 - depths[i] / np.max(depths[i])), cv2.COLOR_GRAY2RGB)

            imgs = [rgb8, normal8, depth8]

            if i < len(controls):
                control = cv2.cvtColor(utils.to8b(controls[i]), cv2.COLOR_RGB2BGR)
                imgs.append(control)

            img8 = cv2.hconcat(imgs)
            img8_th = torch.from_numpy(np.asarray(img8)).to(device)
            img8 = img8_th.detach().cpu().numpy().astype(np.uint8)

            all_img8.append(img8)

        # concat and gather images from all processes for better visualization
        all_img8 = cv2.vconcat(all_img8)
        cv2.imwrite(filename, all_img8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    normals = np.array(normals)

    return rgbs, depths, normals


def seed_everything(args):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    # accelerate.utils.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (cfg_train.pg_scale_factor**len(cfg_train.pg_scale)))

    print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
    if cfg_model.model_type =='dvgo':
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    elif cfg_model.model_type == 'dmtet':
        if cfg_model.coarse_model_type == 'dvgo':
            model_class = dvgo.DirectVoxGO
        else:
            raise Exception('not supported model type: %s' % cfg_model.coarse_model_type)
        coarse_model = utils.load_model(model_class, coarse_ckpt_path)
        model = dvgo_dmtet.DvgoDmtet(coarse_model, grid_res=cfg_model.num_voxels, **cfg_model)
        del coarse_model
    else:
        raise Exception('not supported model type: %s' % cfg_model.model_type)
    return model

def load_existed_model(args, cfg, cfg_model, cfg_train, reload_ckpt_path):
    if cfg_model.model_type == 'dvgo':
        model_class = dvgo.DirectVoxGO
    elif cfg_model.model_type == 'dmtet':
        model_class = dvgo_dmtet.DvgoDmtet
    else:
        raise Exception('not supported model type: %s' % cfg_model.model_type)
    model = utils.load_model(model_class, reload_ckpt_path)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def save_state(global_step, model, optimizer, save_ckpt_path):
    torch.save({
        'global_step': global_step,
        'model_kwargs': model.get_kwargs(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_ckpt_path)


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def scene_rep_reconstruction(args, cfg, accelerator, cfg_model, cfg_train, xyz_min, xyz_max, dataset, stage, coarse_ckpt_path=None, dataset_test=None):

    # init
    device = accelerator.device

    near, far, H, W = dataset.near, dataset.far, dataset.H, dataset.W
    dataloader = DataLoader(dataset, cfg_train.N_img, shuffle=True, drop_last=True,
                            generator=torch.Generator(device=device))

    test_batch_size = 2
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, test_batch_size, shuffle=False, drop_last=False,
                                generator=torch.Generator(device=device))
    else:
        dataloader_test = None
    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    imgs_save_path = os.path.join(cfg.basedir, cfg.expname, 'img_%s' % stage)
    os.makedirs(imgs_save_path, exist_ok=True)
    if args.no_reload:
        reload_ckpt_path = None
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_model, cfg_train, reload_ckpt_path)
        if start >= cfg_train.N_iters:
            return
    model = model.to(device)

    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
        'bg_type': -1
    }
    # diffusion
    human_view = None
    if cfg_model.diffusion.startswith('sd'):
        diffusion_net = StableDiffusion(device, random_sample=True, n_iters=cfg.coarse_train.N_iters,
                                        sd_version=cfg_model.diffusion.split('-')[-1])
        diffusion_net = disable_params_grad(diffusion_net)
    elif cfg_model.diffusion == 'controlnet':

        diffusion_net = create_model('./controlnet/configs/cldm_v15.yaml').cpu()
        control_sig = 'dense_pose'  # pose, dense_pose

        if control_sig == 'pose':
            human_view = HumanPoseSMPLX('./human/smpl', crop_mode=cfg_model.avatar_type)
            diffusion_net.load_state_dict(load_state_dict(cfg_model.diffusion_path, location=device))
            if args.mixed_precision == 'fp16':
                diffusion_net = diffusion_net.to(torch.float16)
            verts, faces = human_view.get_mesh()
            print(verts.shape, faces.shape)
            if hasattr(model, 'init_density') and reload_ckpt_path is None:
                model.init_density(verts, faces)
        elif control_sig == 'dense_pose':
            diffusion_net.load_state_dict(load_state_dict(cfg_model.diffusion_path, location=device))
            if args.mixed_precision == 'fp16':
                diffusion_net = diffusion_net.to(torch.float16)

            human_view = DensePoseSMPL('./human/smpl', crop_mode=cfg_model.avatar_type, pose_type=dataset.pose_type)
            verts, faces = human_view.get_mesh()
            if hasattr(model, 'init_density') and reload_ckpt_path is None:
                model.init_density(verts, faces)

            if cfg_model.model_type == 'dmtet':
                diffusion_net.max_step = int(diffusion_net.num_train_timesteps * cfg_train.tmax)
                print("Setting diffusion max_timestep to", cfg_train.tmax)
        else:
            raise Exception('not supported sontrol signal: %s' % control_sig)
        diffusion_net = diffusion_net.to(accelerator.device)
        diffusion_net = disable_params_grad(diffusion_net)

    else:
        raise Exception('not supported diffusion type!')

    negative_prompt = '' if not hasattr(cfg.data, 'negative_text') else cfg.data.negative_text
    with torch.no_grad():
        text_zs = prepare_text_embeddings(diffusion_net, cfg.data.text, dir_text=cfg.data.dir_text, negative=negative_prompt)

    model = prepare_model(model, accelerator, cfg_model)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    iter_train = iter(dataloader)
    if dataloader_test is not None:
        iter_test = iter(dataloader_test)
    else:
        iter_test = None

    # set radius and focus_mode status when restore from pretrained process
    for global_step in trange(1, start):
        if global_step in cfg_train.radius_stage:
            dataloader.dataset.set_radius(cfg_train.radius_stage.index(global_step) + 1)

        if human_view is not None and hasattr(cfg_train, 'focus_start_iter') and global_step >= int(cfg_train.focus_start_iter):
            dataloader.dataset.set_focus(human_view.focus)

    # GOGO
    torch.cuda.empty_cache()
    time0 = time.time()
    global_step = -1
    guidance_scale = cfg_train.guidance
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        if cfg_model.model_type != 'dmtet':
            
            # renew occupancy grid
            if model.mask_cache is not None and (global_step + cfg_train.mask_step//2) % cfg_train.mask_step == 0:
                model.update_occupancy_cache()
            # progress scaling checkpoint
            if global_step in cfg_train.pg_scale:
                n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
                cur_voxels = int(cfg_model.num_voxels / (cfg_train.pg_scale_factor**n_rest_scales))

                model = unwarp_model(model, accelerator, cfg_model)
                model.scale_volume_grid(cur_voxels)
                optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=global_step)
                optimizer = accelerator.prepare(optimizer)
                model = prepare_model(model, accelerator, cfg_model)
                torch.cuda.empty_cache()
                
            if global_step in cfg_train.tighten_bbox:
                model = unwarp_model(model, accelerator, cfg_model)
                xyz_min, xyz_max = model.tight_bbox_for_coarse_world(
                    cfg_train.tighten_thresh[cfg_train.tighten_bbox.index(global_step)])
                optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=global_step)
                optimizer = accelerator.prepare(optimizer)
                model = prepare_model(model, accelerator, cfg_model)

        if global_step in cfg_train.radius_stage:
            dataloader.dataset.set_radius(cfg_train.radius_stage.index(global_step) + 1)

        if human_view is not None and hasattr(cfg_train, 'focus_start_iter') and global_step >= int(cfg_train.focus_start_iter):
            dataloader.dataset.set_focus(human_view.focus)

        poses, dir_texts, light_d, Ks = next(iter_train)

        if global_step % (len(dataset) // cfg_train.N_img) == 0:
            iter_train = iter(dataloader)

        text_dir, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_diffusion_rays(
            rgb_tr=dir_texts, train_poses=poses, HW=(H, W), Ks=Ks,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        target = text_zs[text_dir]
        # guidance for different views
        guidance_scale_tensor = torch.tensor([guidance_scale] * len(text_dir))
        # print(target.shape, sel_b, rgb_tr.shape)
        rays_o = rays_o_tr.reshape([-1, 3])
        rays_d = rays_d_tr.reshape([-1, 3])
        viewdirs = viewdirs_tr.reshape([-1, 3])

        control = None
        control_org = None
        if human_view:
            control = human_view.render(poses, Ks, (cfg.data.height, cfg.data.width), vis=False, out_type=control_sig)
            control_org = control

        if global_step == 1:
            if dataset_test is None:
                if control_org is not None:
                    control_psuedo = control_org.reshape(cfg_train.N_img, 3, cfg.data.height, cfg.data.width)
                    dataset_train_vis = PsuedoDataset(H, W, [poses, poses, light_d, Ks, control_psuedo], num=5)
                else:
                    dataset_train_vis = PsuedoDataset(H, W, [poses, poses, light_d, Ks], num=5)
            else:
                poses_test, dir_texts_test, light_d_test, Ks_test = next(iter_test)

                if global_step % (len(dataset_test) // test_batch_size*args.i_print) == 0:
                    iter_test = iter(dataloader_test)
                dataset_train_vis = PsuedoDataset(H, W, [poses_test, poses_test, light_d_test, Ks_test], num=test_batch_size)
                render_kwargs['bg_type'] = 0.5

            render_viewpoints(cfg, cfg_model, model, dataset_train_vis, cfg.data.ndc, render_kwargs,
                              shading='albedo', savedir=imgs_save_path, dump_images=True,
                              prefix='iter_{:04d}_'.format(global_step-1), render_factor=H/512.)

        if global_step < cfg_train.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            ambient_ratio = 0.3
            rand_ = random.random()
            if rand_ < 0.4:
                shading = 'lambertian'
            elif rand_ < 0.5:
                shading = 'lambertian'
            else:
                shading = 'albedo'
                ambient_ratio = 1.0

        bg_type = -1
        rand_bg = np.random.random()
        if rand_bg > 0.7:
            bg_type = 0.
        elif rand_bg > 0.4:
            bg_type = 1.
        render_kwargs['bg_type'] = bg_type

        if cfg_model.model_type != 'dmtet':
            light_d = light_d.reshape([cfg_train.N_img, 1, 1, 3]).repeat([1, H, W, 1]).reshape([-1, 3])
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, ld, shading, ambient_ratio, **render_kwargs).items()}
                for ro, rd, vd, ld in zip(rays_o.split(cfg_train.N_rand, 0), rays_d.split(cfg_train.N_rand, 0),
                                    viewdirs.split(cfg_train.N_rand, 0), light_d.split(cfg_train.N_rand, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks])
                for k in render_result_chunks[0].keys()
            }
        else:
            light_d = light_d.reshape([cfg_train.N_img, 3])
            render_result = model(Ks, poses, viewdirs, H, W, light_d, shading, ambient_ratio, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad()
        img = torch.reshape(render_result['rgb_marched'], [cfg_train.N_img, H, W, -1]).permute(0, 3, 1, 2).contiguous()
        
        # backward finished
        with accelerator.autocast():
            if control is not None and global_step == 1:
                if hasattr(diffusion_net, 'sample_img'):
                    sample_save_dir = os.path.join(cfg.basedir, cfg.expname, 'samples')
                    os.makedirs(sample_save_dir, exist_ok=True)
                    res = diffusion_net.sample_img(target, control)
                    # control + rgb
                    for img_i in range(len(res) // 2):
                        control8 = cv2.cvtColor(res[img_i * 2], cv2.COLOR_RGB2BGR)
                        rgb8 = cv2.cvtColor(res[img_i * 2 + 1], cv2.COLOR_RGB2BGR)
                        img8 = cv2.hconcat([control8, rgb8])
                        # img8 = (control8*0.2 + rgb8*0.8).astype(np.uint8)
                        cv2.imwrite(os.path.join(sample_save_dir, f'./step_{global_step}_{img_i}.jpg'), img8)
            loss = diffusion_net.train_step(target, img, latent_img=cfg_model.latent,
                                            guidance_scale=guidance_scale_tensor, n_sample=1,
                                            control_hint=control, accelerator=accelerator)


        loss_smooth = torch.scalar_tensor(0.)
        if 'loss_smooth' in render_result and global_step > cfg_train.smooth_iters:
            loss_smooth = torch.mean(render_result['loss_smooth']) * cfg_train.weight_loss_smooth
            # loss_smooth.backward(retain_graph=True)
            accelerator.backward(loss_smooth, retain_graph=True)
            loss += loss_smooth
            # print('smooth loss')

        optimizer.step()

        # update lr
        max_lr = -1
        relative_decay_factor = utils.lr_decay_func(global_step, cfg_train) / utils.lr_decay_func(max(0, global_step-1), cfg_train)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * relative_decay_factor
            if param_group['lr'] > max_lr:
                max_lr = param_group['lr']

        # check log & save
        if global_step%args.i_print==0:
            if stage == 'coarse':
                grid_usage_rate = torch.mean(model.mask_cache.mask.to(torch.float)).item()
            else:
                grid_usage_rate = 1.
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            # if accelerator.is_main_process:
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                    f'guidance: {guidance_scale:.1f} / '
                    f'Loss: {loss.item()*100:.3f} / '
                    f'l_smooth: {loss_smooth.item()*100:.3f} / '
                    f'lr_density: {max_lr:.5f} / '
                    f'grid_usage_rate: {grid_usage_rate:.3f} / '
                    f'Eps: {eps_time_str}')

            if dataset_test is None:
                if control_org is not None:
                    control_psuedo = control_org.reshape(cfg_train.N_img, 3, *control_org.shape[-2:])
                    dataset_train_vis = PsuedoDataset(H, W, [poses, poses, light_d, Ks, control_psuedo], num=5)
                else:
                    dataset_train_vis = PsuedoDataset(H, W, [poses, poses, light_d, Ks], num=5)
            else:
                print(len(dataset_test),  test_batch_size, global_step)
                if global_step % (len(dataset_test) // test_batch_size * args.i_print) == 0:
                    iter_test = iter(dataloader_test)
                    # print('new iter')
                poses_test, dir_texts_test, light_d_test, Ks_test = next(iter_test)

                dataset_train_vis = PsuedoDataset(H, W, [poses_test, poses_test, light_d_test, Ks_test], num=test_batch_size)
                shading = 'albedo'
                render_kwargs['bg_type'] = 0.5
                # print('test_render')

            render_viewpoints(cfg, cfg_model, model, dataset_train_vis, cfg.data.ndc, render_kwargs,
                              shading=shading, savedir=imgs_save_path, dump_images=True,
                              prefix='iter_{:04d}_'.format(global_step), render_factor=H/512.)

        if global_step%args.i_weights == 0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
            model = unwarp_model(model, accelerator, cfg_model)
            save_state(global_step, model, optimizer, path)
            model = prepare_model(model, accelerator, cfg_model)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
        model = unwarp_model(model, accelerator, cfg_model)
        save_state(global_step, model, optimizer, last_ckpt_path)
        model = prepare_model(model, accelerator, cfg_model)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

    del model
    del diffusion_net


def train(args, cfg, accelerator, dataset_train, dataset_train_fine, dataset_test=None):
    device = accelerator.device
    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    try:
        cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))
    except:
        pass
    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = torch.tensor([-1, -1., -1], dtype=torch.float, device=device), torch.tensor([1, 1., 1], dtype=torch.float, device=device)

    if cfg.coarse_train.N_iters > 0:
        if dataset_test is not None:
            dataset_test.pose_type = 'nerf'

        scene_rep_reconstruction(
                args=args, cfg=cfg, accelerator=accelerator,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                dataset=dataset_train, stage='coarse', dataset_test=dataset_test)
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    if cfg.fine_train.N_iters <= 0:
        eps_time = time.time() - eps_time
        eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
        print('train: finish (eps time', eps_time_str, ')')
        return

    # fine detail reconstruction
    eps_fine = time.time()
    accelerator.free_memory()
    torch.cuda.empty_cache()
    xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    if dataset_test is not None:
        dataset_test.pose_type = 'tetra'

    scene_rep_reconstruction(
            args=args, cfg=cfg, accelerator=accelerator,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            dataset=dataset_train_fine, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path, dataset_test=dataset_test)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


def pipeline(args, cfg):
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision)
    device = accelerator.device
    data_config = copy.deepcopy(cfg.data)
    dataset_train = Diffusion3dDataset(training=True, device=device, args=data_config)
    dataset_train_fine = Diffusion3dDataset(training=True, device=device, args=cfg.data_fine)

    test_cfg = cfg.data.copy()
    test_cfg.height = 1024
    test_cfg.width = 1024
    dataset_test = Diffusion3dDataset(training=False, device=device, args=test_cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = torch.tensor([-1., -1., -1.], dtype=torch.float, device=device), \
            torch.tensor([1., 1., 1.], dtype=torch.float, device=device)

        near, far = dataset_train.near, dataset_train.far
        cam_lst = []
        for i in range(len(dataset_train)):
            c2w, _, _, K = dataset_train[i]
            H, W = dataset_train.H, dataset_train.W
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, )
            cam_o = rays_o[0, 0].cpu().numpy()
            cam_d = rays_d[[0, 0, -1, -1], [0, -1, 0, -1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o + cam_d * max(near, far * 0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
                            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
                            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, accelerator, dataset_train, dataset_train_fine, dataset_test=None)

    accelerator.free_memory()
    torch.cuda.empty_cache()
    # load model for rendring
    if args.render_test or args.export_mesh:
        model_config = {
            'coarse': cfg.coarse_model_and_render,
            'fine': cfg.fine_model_and_render,
        }
        train_config = {
            'coarse': cfg.coarse_train,
            'fine': cfg.fine_train,
        }

        for stage in (['coarse', 'fine'] if not args.render_fine_only else ['fine']):
            cur_model_config = model_config[stage]
            cur_train_config = train_config[stage]
            print(cfg.basedir, cfg.expname, stage)
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, '%s_last.tar' % stage)
            ckpt_name = ckpt_path.split('/')[-1][:-4]

            if cur_model_config.model_type == 'dvgo':
                model_class = dvgo.DirectVoxGO
            elif cur_model_config.model_type == 'dmtet':
                model_class = dvgo_dmtet.DvgoDmtet
            else:
                raise Exception('not supported model type: %s' % cfg.coarse_model_and_render.model_type)
            if not os.path.exists(ckpt_path):
                continue
            model = utils.load_model(model_class, ckpt_path).to(device)
            stepsize = cur_model_config.stepsize
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': dataset_train.near,
                    'far': dataset_train.far,
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_depth': True,
                    'bg_type': 0.5,
                    'filter': True
                },
            }
            if stage == 'coarse':
                dataset_test.pose_type = 'nerf'
                dataset_train.pose_type = 'nerf'
            else:
                dataset_test.pose_type = 'tetra'
                dataset_train.pose_type = 'tetra'
                dataset_test.radius_range *= 1.1  # dirty fix, unclear why fine stage tends to render larger result

            # render testset and eval
            if args.render_test:
                if hasattr(args, 'export_path'):
                    testsavedir = os.path.join(args.export_path, f'render_test_{ckpt_name}')
                else:
                    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
                os.makedirs(testsavedir, exist_ok=True)

                rgbs, depths, normals = render_viewpoints(cfg, cur_model_config,
                                                          dataset=dataset_test, savedir=testsavedir,
                                                          dump_images=args.dump_images, **render_viewpoints_kwargs)
                print('All results are dumped into', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=10, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=10, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, 'video.normal.mp4'), utils.to8b(normals/2.+0.5), fps=10, quality=8)

        if args.export_mesh:
            if hasattr(args, 'export_path'):
                model_path = os.path.join(args.export_path, 'export_mesh')
            else:
                model_path = os.path.join(cfg.basedir, cfg.expname, 'export_mesh')
            os.makedirs(model_path, exist_ok=True)
            cur_model_config = cfg.fine_model_and_render
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')

            if cur_model_config.model_type == 'dvgo':
                model_class = dvgo.DirectVoxGO
            elif cur_model_config.model_type == 'dmtet':
                model_class = dvgo_dmtet.DvgoDmtet
            else:
                raise Exception('not supported model type: %s' % cfg.coarse_model_and_render.model_type)
            if os.path.exists(ckpt_path):
                model = utils.load_model(model_class, ckpt_path).to(device)
                model.extract_3d_shape(model_path)
                print('export obj done')

    print('Done')