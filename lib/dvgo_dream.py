from mimetypes import init
import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from torch_scatter import segment_coo

from . import grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
try:
    render_utils_cuda = load(
            name='render_utils_cuda',
            sources=[
                os.path.join(parent_dir, path)
                for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
            verbose=True)
except:
    render_utils_cuda = None

def act(x, act_type):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)(x)
    elif act_type == 'leaky_relu':
        act(nn.LeakyReLU(negative_slope=0.01, inplace=True)(x) + 0.1, act_type='relu')
    else:
        raise Exception('not support act type')

'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, num_bg_voxels=0,
                 mask_cache_world_size=None,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4, latent=False, org_xyz_min=None, org_xyz_max=None,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        if org_xyz_min is None:
            org_xyz_min = xyz_min
        if org_xyz_max is None:
            org_xyz_max = xyz_max
        self.org_xyz_min = torch.Tensor(org_xyz_min)
        self.org_xyz_max = torch.Tensor(org_xyz_max)
        self.fast_color_thres = fast_color_thres
        self.bound_radius = (self.xyz_max - self.xyz_min).amax() / 2.
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        
        # determine the density bias shift
        self.act_softplus = True
        if self.act_softplus:
            # self.register_buffer('act_shift', torch.FloatTensor([init_ball_value]))
            self.register_buffer('act_shift', torch.FloatTensor([0.]))
            print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, num_bg_voxels)

        # # init density voxel grid
        self.density = grid.create_grid(
            'DenseGrid', channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            init_method='constant', init_mean=1e-4, init_std=0.5, add_blob=True, alpha=True, bias=True)
        # init_method='normal', init_mean=0, init_std=1., add_blob=True)
        # self.init_grid_with_cylinder(radius=0.5, value=init_ball_value, density=True)
        # self.init_grid_with_cylinder(radius=0.4, value=1e7, density=True)
        # self.init_grid_with_sphere(radius=0.2, value=0., density=True)
        # self.init_grid_with_gaussian(mean=1., std=0.1, density=True, add=False)
        # self.init_grid_blob(mean=10., radius=0.5)
        # self.init_grid_with_sphere(radius=0.7, value=1e7, density=True)

        self.normal_scale = 1./ 2
        normal_world_size = (self.world_size*self.normal_scale).long()
        # normal_world_size = torch.tensor([4,4,4])
        # self.density_normal = grid.create_grid(
        #         'DenseGrid', channels=3, world_size=normal_world_size,
        #         xyz_min=self.xyz_min, xyz_max=self.xyz_max,
        #         init_method='constant', init_mean=0., init_std=0.5, add_outward=True)
                # init_method='normal', init_mean=0, init_std=1.)
        # self.init_normal_inner_out()
        # self.density_normal = None
        # self.density_blob = grid.create_grid(
        #         density_type, channels=1, world_size=self.world_size * self.voxel_size_ratio.long(),
        #         xyz_min=self.xyz_min, xyz_max=self.xyz_max,
        #         init_method='constant', init_mean=0., init_std=0.5)
        # self.init_grid_blob(mean=10., radius=0.5)
        # init color representation
        self.densitynet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }

        # self.img_2d = nn.Parameter(torch.randn(1, 3, 64, 64) / 2. + 0.5)
        self.img_2d = None
        self.latent = latent
        self.color_dim = 4 if latent else 3

        # color voxel grid  (coarse stage)
        # self.k0_dim = self.color_dim
        self.k0_dim = rgbnet_dim if rgbnet_dim > 0 else self.color_dim
        init_mean = 0. if latent else 0.5
        self.k0 = grid.create_grid(
            'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            init_method='constant', init_mean=0., init_std=0.5, bias=True)
        self.k0.bias = grid.create_grid(
                'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                init_method='normal', init_mean=init_mean, init_std=1,
                add_blob=False, add_outward=False, alpha=False, bias=False)
        self.k0.bias.grid.requires_grad = False
        if rgbnet_dim > 0:
            # self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            # dim0 = (3 + 3 * viewbase_pe * 2)
            # dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(self.k0_dim, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth - 2)
                ],
                nn.Linear(rgbnet_width, self.color_dim)
            )
            for layer in range(len(self.rgbnet)):
                nn.init.constant_(self.rgbnet[layer].bias, 0)
        else:
            self.rgbnet = None
        
        # background color
        # self.background = grid.MLPBackground(4, 128, 3, self.color_dim)
        # self.background = nn.Parameter(torch.ones([3,3]))

        self.background = grid.create_grid(
                'SphereGrid', channels=self.color_dim, world_size=self.num_bg_voxels,
                # init_method='constant', init_mean=0.9, init_std=0.0166)
                init_method='normal', init_mean=init_mean, init_std=1.)
        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
            
        mask = torch.ones(list(mask_cache_world_size)).bool()
        # print(torch.sum(d>1), torch.sum(d>-1), d.shape, torch.min(self_grid_xyz), torch.max(self_grid_xyz))
        self.mask_cache = grid.MaskGrid(mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max, 
                # bound_radius=self.bound_radius
                )

        self.init_gradient_conv()

    @torch.no_grad()
    def init_density(self, verts, faces, path=None):
        # TODO xyz_min and xyz_max由于tighten 变化了，导致模型restore之后和最初的计算值不同
        if verts is None:
            # init_obj = './human/head/mean_head3d_uv_watertight.obj'
            init_obj = path
        else:
            init_obj = None

        dim_ratio = torch.tensor(self.org_xyz_max - self.org_xyz_min).cpu().numpy()
        if faces is not None:
            # init_obj = './data/sdf/bunny_watertight.obj'
            world_size_bias = self.density.bias.world_size
            # print(world_size_numpy)
            density_bias, _ = init_voxel(init_obj, world_size_bias,
                                         torch.tensor(self.org_xyz_min),
                                         torch.tensor(self.org_xyz_max),
                                         value=5., ratio=0.6, verts=verts, faces=faces)
            self.density.bias = density_bias
        verts_max = np.minimum(np.max(verts, axis=0), torch.tensor(self.org_xyz_max).cpu().numpy())
        verts_min = np.maximum(np.min(verts, axis=0), torch.tensor(self.org_xyz_min).cpu().numpy())
        xyz_range = verts_max - verts_min
        xyz_range = xyz_range * dim_ratio
        scale_v = min(max(xyz_range[1] / (xyz_range[0] + 1e-3), xyz_range[1] / (xyz_range[2] + 1e-3)), 5.)
        # scale = torch.tensor([min(xyz_range[1] / (xyz_range[0] + 1e-3), 5.), 1., min(xyz_range[1] / (xyz_range[2] + 1e-3), 5.)], dtype=torch.float)
        scale = torch.tensor([scale_v, 1.3, scale_v], dtype=torch.float)

        print("density blob scale", scale, xyz_range)
        # scale = torch.tensor([1.8, 1., 1.8], dtype=torch.float)
        self.density.set_blob_para(scale=scale)
    #
    @torch.no_grad()
    def init_grid_with_gaussian(self, mean=5., std=0.2, density=True, add=False):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = (torch.stack(torch.meshgrid(
            torch.linspace(0, self.world_size[0] - 1, self.world_size[0]),
            torch.linspace(0, self.world_size[1] - 1, self.world_size[1]),
            torch.linspace(0, self.world_size[2] - 1, self.world_size[2]),
        ), -1).float()) / (self.world_size - 1) - 0.5
        nearest_dist = torch.sum(self_grid_xyz * self_grid_xyz, dim=-1)[None, None, ...]
        value = mean * torch.exp(-nearest_dist / std / std / 2.)

        if density:
            if add:
                self.density.grid[:] += value
            else:
                self.density.grid[:] = value
        else:
            if add:
                self.k0.grid[:] += value
            else:
                self.k0.grid[:] = value
        # threshhold = mean * np.exp(-0.25 / std / std / 2.)
        # self.density.grid[self.density.grid < threshhold] = -1
        threshhold = -100.

        # softplus init
        if self.act_softplus:
            mask = self.density.grid >= threshhold
            mask_value = self.density.grid[mask]
            self.density.grid[mask] = torch.log(torch.exp(mask_value) - 1.) - self.act_shift.to(
                self.density.grid.device)

    def _set_grid_resolution(self, num_voxels, num_bg_voxels):
        # Determine grid resolution
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        self.num_bg_voxels = num_bg_voxels
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)
        print('dvgo: voxel_background_size', self.num_bg_voxels)

    def init_img(self, diffusion_model, model_type, size):
        if self.img_2d is None:
            if 'sd' in model_type:
                self.img_2d = nn.Parameter(diffusion_model.decode_latents(torch.randn(1, 4, size//8, size//8)))
            else:
                self.img_2d = nn.Parameter(torch.randn(1, 3, size, size) / 2. + 0.5)
    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'org_xyz_min': self.org_xyz_min.cpu().numpy(),
            'org_xyz_max': self.org_xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'num_bg_voxels': self.num_bg_voxels,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'latent': self.latent,
            **self.densitynet_kwargs,
        }
    
    @torch.no_grad()
    def init_normal_inner_out(self):
        # maskout grid points that between cameras and their near planes
        normal_world_size = (self.normal_scale * self.world_size).cpu().numpy().astype(np.int)
        self_grid_xyz = (torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], normal_world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], normal_world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], normal_world_size[2]),
        ), -1).float()).flip((-1,))

        dir = self_grid_xyz / (torch.sqrt(torch.mean(self_grid_xyz*self_grid_xyz, dim=-1, keepdim=True)) + 1e-7)
        value = torch.permute(dir, [3, 0, 1, 2])[None, ...]
        self.density_normal.grid[:] = value

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, num_bg_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        ori_num_bg_voxels = self.num_bg_voxels
        self._set_grid_resolution(num_voxels, num_bg_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())
        print('dvgo: scale_volume_grid scale bg_size from', ori_num_bg_voxels, 'to', self.num_bg_voxels)

        self.density.scale_volume_grid(self.world_size)
        # self.density_normal.scale_volume_grid((self.world_size*self.normal_scale).long())
        self.k0.scale_volume_grid(self.world_size)
        # self.background.scale_volume_grid(self.num_bg_voxels)

        # if np.prod(self.world_size.tolist()) <= 64**3:
        #     self_grid_xyz = torch.stack(torch.meshgrid(
        #         torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
        #         torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
        #         torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        #     ), -1)
        #     self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
        #     self.mask_cache = grid.MaskGrid(
        #             mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
        #             xyz_min=self.xyz_min, xyz_max=self.xyz_max, bound_radius=self.bound_radius)
        
        torch.cuda.empty_cache()
        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        # print('test shape ', cache_grid_xyz.shape, cache_grid_density.shape)
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

        # # methods to optimize the empty inner model
        # cache_grid_density = F.max_pool3d(cache_grid_density, kernel_size=3, padding=1, stride=1)

    def init_gradient_conv(self, sigma=0):
        self.grad_conv = nn.Conv3d(1,3,(3,3,3),stride=(1,1,1), padding=(1, 1, 1), padding_mode='replicate')
        kernel = np.asarray([
            [[1,2,1],[2,4,2],[1,2,1]],
            [[2,4,2],[4,8,4],[2,4,2]],
            [[1,2,1],[2,4,2],[1,2,1]],
        ])
        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i,j,k] = ((i-1)**2 + (j-1)**2 + (k-1)**2 - 1)
        kernel0 = kernel * np.exp(-distance * sigma)

        kernel1 = kernel0 / (kernel0[0].sum() * 2 * self.voxel_size.item())
        weight = torch.from_numpy(np.concatenate([kernel1[None] for _ in range(3)])).float()
        weight[0,1,:,:] *= 0
        weight[0,0,:,:] *= -1
        weight[1,:,1,:] *= 0
        weight[1,:,0,:] *= -1
        weight[2,:,:,1] *= 0
        weight[2,:,:,0] *= -1
        # self.grad_conv.weight.data = weight.unsqueeze(1).float()
        # self.grad_conv.bias.data = torch.zeros(3)
        # for param in self.grad_conv.parameters():
        #     param.requires_grad = False

        # smooth conv for TV
        self.tv_smooth_conv = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, padding_mode='replicate')
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.tv_smooth_conv.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.tv_smooth_conv.bias.data = torch.zeros(1)
        for param in self.tv_smooth_conv.parameters():
            param.requires_grad = False

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def density_normal_derive_tv_loss(self):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        gradient = torch.zeros([1, 3] + [*self.density.grid.shape[-3:]]).to(self.density.grid.device)
        density = self.density(self_grid_xyz).reshape([1, 1] + [*self.density.grid.shape[-3:]])
        gradient[:, 0, 1:-1, :, :] = (density[:, 0, 2:, :, :] - density[:, 0, :-2, :, :]) / 2 / self.voxel_size
        gradient[:, 1, :, 1:-1, :] = (density[:, 0, :, 2:, :] - density[:, 0, :, :-2, :]) / 2 / self.voxel_size
        gradient[:, 2, :, :, 1:-1] = (density[:, 0, :, :, 2:] - density[:, 0, :, :, :-2]) / 2 / self.voxel_size

        nonempty_mask = self.mask_cache(self_grid_xyz)[None, None].contiguous()
        smooth_tv_error = (
                    self.tv_smooth_conv(gradient.permute(1, 0, 2, 3, 4)).detach() - gradient.permute(1, 0, 2, 3, 4))
        smooth_tv_error = smooth_tv_error[nonempty_mask.repeat(3, 1, 1, 1, 1)] ** 2
        # tv = smooth_tv_error.mean()

        return smooth_tv_error

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        
        if self.act_softplus:
            return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)
        else:
            return 1.- torch.exp(-torch.nn.ReLU()(density) * interval)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)
    
    @torch.no_grad()
    def tight_bbox_for_coarse_world(self, thres):
        print('compute_bbox_by_coarse_geo: start')
        eps_time = time.time()
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.world_size[0]),
            torch.linspace(0, 1, self.world_size[1]),
            torch.linspace(0, 1, self.world_size[2]),
        ), -1)
        dense_xyz = self.xyz_min * (1-interp) + self.xyz_max * interp
        density = self.density(dense_xyz)
        alpha = self.activate_density(density)

        # color = self.k0(dense_xyz).amax(-1)
        mask = (alpha > thres) # & (color > 1e-2)

        active_xyz = dense_xyz[mask]
        tight_xyz_min = active_xyz.amin(0)
        tight_xyz_max = active_xyz.amax(0)

        world_range = (tight_xyz_max - tight_xyz_min) * 1.2
        world_center = (tight_xyz_max + tight_xyz_min) * 0.5
        tight_xyz_min = torch.maximum(world_center - world_range / 2., self.xyz_min)
        tight_xyz_max = torch.minimum(world_center + world_range / 2., self.xyz_max)
        print('tight_bbox_for_coarse_world: xyz_min', tight_xyz_min)
        print('tight_bbox_for_coarse_world: xyz_max', tight_xyz_max)

        # set to tight world 
        # variable
        self.xyz_min = tight_xyz_min
        self.xyz_max = tight_xyz_max
        # self.xyz_min /= 2.
        # self.xyz_max /= 2.
        self._set_grid_resolution(self.num_voxels, self.num_bg_voxels)
        # density resize
        self.density.crop_and_scale_volume_grid(self.xyz_min, self.xyz_max, self.world_size)
        # k0 resize
        self.k0.crop_and_scale_volume_grid(self.xyz_min, self.xyz_max, self.world_size)
        # mask resize
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.world_size[0]),
            torch.linspace(0, 1, self.world_size[1]),
            torch.linspace(0, 1, self.world_size[2]),
        ), -1)
        tight_xyz = self.xyz_min * (1-interp) + self.xyz_max * interp
        tmp = self.mask_cache(tight_xyz)
        # tmp = torch.ones(list(self.world_size), dtype=torch.bool)

        self.mask_cache = grid.MaskGrid(tmp, self.xyz_min, self.xyz_max, bound_radius=self.bound_radius)
        # normal resize
        normal_world_size = (self.world_size*self.normal_scale).long()
        # self.density_normal.crop_and_scale_volume_grid(self.xyz_min, self.xyz_max, normal_world_size)
        eps_time = time.time() - eps_time
        print('tight_bbox_for_coarse_world: finish (eps time:', eps_time, 'secs)')
        return tight_xyz_min, tight_xyz_max

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, light_d, shading='lambertian', ambient_ratio=0.1, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        bg_type = render_kwargs['bg_type']

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        # query for color
        k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # rgb = torch.sigmoid(k0)
            rgb = k0
        else:
            # viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            # viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            # viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            # rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(k0)
            # rgb = torch.sigmoid(rgb_logit)
            rgb = rgb_logit

        ray_pts_light = light_d[ray_id]
        rgb, normal = self.render(ray_pts, rgb, ray_pts_light, shading, ambient_ratio, **render_kwargs)
        normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        if normal is not None:  # and shading != 'albedo':
            # orientation loss
            # # rays_d_pts = rays_d[ray_id]
            # loss_orient = weights.detach() * ((normal * rays_d[ray_id]).sum(dim=-1).clamp(min=0) ** 2)
            # ret_dict['loss_orient'] = loss_orient

            loss_tv_grad = self.density_normal_derive_tv_loss()
            ret_dict['loss_smooth'] = loss_tv_grad
        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, self.color_dim]),
                reduce='sum')
        
        # if shading == 'textureless':
        #     # bg_colors = 0.
        #     bg_colors = self.background(viewdirs)
        # else:
        if bg_type < 0:
            bg_colors = self.background(viewdirs)
        else:
            bg_colors = max(min(bg_type, 1.), 0.)
        # rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        rgb_marched += (alphainv_last.unsqueeze(-1) * bg_colors)
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'normal_marched': normal_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })
        
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict

    def common_forward(self, ray_pts):
        # interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        density = self.density(ray_pts, smooth=False)
        # sigma = self.activate_density(density, interval)
        # sigma = F.softplus(density + self.act_shift)
        return density

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos = self.common_forward(
            (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(self.xyz_min[0], self.xyz_max[0]))
        dx_neg = self.common_forward(
            (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(self.xyz_min[0], self.xyz_max[0]))
        dy_pos = self.common_forward(
            (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(self.xyz_min[1], self.xyz_max[1]))
        dy_neg = self.common_forward(
            (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(self.xyz_min[1], self.xyz_max[1]))
        dz_pos = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(self.xyz_min[2], self.xyz_max[2]))
        dz_neg = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(self.xyz_min[2], self.xyz_max[2]))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def finite_difference_normal_2(self, x, epsilon=1e-2):
        # x: [N, 3]
        pixel_max_len = (self.xyz_max-self.xyz_min).amax(dim=0)/64. * 2
        max_epsilon = max(max(epsilon, 2*self.voxel_size), pixel_max_len.item())

        sample_num = 10
        sample_epsilon = torch.rand(len(x), sample_num, 1) * max_epsilon + 1e-3
        x_sample = x.unsqueeze(1)
        x_pos = x_sample + torch.cat([sample_epsilon, torch.zeros(len(x), sample_num, 2)], dim=-1)
        x_neg = x_sample - torch.cat([sample_epsilon, torch.zeros(len(x), sample_num, 2)], dim=-1)

        z_pos = x_sample + torch.cat([torch.zeros(len(x), sample_num, 2), sample_epsilon], dim=-1)
        z_neg = x_sample - torch.cat([torch.zeros(len(x), sample_num, 2), sample_epsilon], dim=-1)

        y_pos = x_sample + torch.cat([torch.zeros(len(x), sample_num, 1), sample_epsilon, torch.zeros(len(x), sample_num, 1)], dim=-1)
        y_neg = x_sample - torch.cat([torch.zeros(len(x), sample_num, 1), sample_epsilon, torch.zeros(len(x), sample_num, 1)], dim=-1)
        dx_pos = self.density(x_pos)
        dx_neg = self.density(x_neg)
        dy_pos = self.density(y_pos)
        dy_neg = self.density(y_neg)
        dz_pos = self.density(z_pos)
        dz_neg = self.density(z_neg)

        sample_epsilon = sample_epsilon.squeeze(-1)
        sample_normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / sample_epsilon,
            0.5 * (dy_pos - dy_neg) / sample_epsilon,
            0.5 * (dz_pos - dz_neg) / sample_epsilon
        ], dim=-1)
        normal = torch.mean(sample_normal, dim=1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        # normal = self.finite_difference_normal_2(x)
        # normal = self.density_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal

    def render(self, ray_pts, albedo, light_d, shading, ambient_ratio, **render_kwargs):
        '''
        ray_pts: Nx3
        albedo: Nx3
        light_d: Nx3 light position
        '''
        if shading == 'albedo':
            # no need to query normal
            # normal = None
            normal = self.normal(ray_pts)
            color = albedo
        else:
            # query normal
            normal = self.normal(ray_pts)
            # lambertian shading
            l = safe_normalize(light_d - ray_pts) # [N, 3]
            lambertian = ambient_ratio + (1 - ambient_ratio) * torch.einsum('ij,ij->i', [normal, l]).clamp(min=0).unsqueeze(-1) # [N, 1]
            
            if shading == 'textureless':
                color = lambertian.repeat(1, self.color_dim)
                # color = torch.cat([normal, lambertian*2.-1], dim=1)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian
        return color, normal


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def init_voxel(mesh_path, reso, xyz_min, xyz_max, value=1., ratio=0.8, verts=None, faces=None):
    # print(xyz_max, xyz_min, (xyz_max - xyz_min).amax())
    # mesh_size = (xyz_max - xyz_min).amax().item() / ratio

    xx, yy, zz = voxelize_mesh_2(mesh_path, reso, ratio, verts, faces, xyz_min.cpu().numpy(), xyz_max.cpu().numpy())
    # print(xx.shape)
    density_bias = grid.create_grid(
        'DenseGrid', channels=1, world_size=reso,
        xyz_min=xyz_min, xyz_max=xyz_max,
        init_method='constant', init_mean=0., init_std=0.5,
        add_blob=False, add_outward=False, alpha=False, bias=False)
    density_bias.grid.data[:, :, xx, yy, zz] = value
    density_bias.grid.requires_grad = False
    xyz_range = [np.max(xx) - np.min(xx), np.max(yy) - np.min(yy), np.max(zz) - np.min(zz)]
    return density_bias, xyz_range
    # grid.density_data.data[target_ind.long()] = value


def voxelize_mesh_2(mesh_path=None, reso=None, ratio=None, verts=None, faces=None, xyz_min=None, xyz_max=None):
    # voxel_grid_path = mesh_path.replace('.obj', '_voxel_grid.npy')
    # if not os.path.exists(voxel_grid_path):
    if mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound

        max_size = np.max(max_bound - min_bound) / ratio
        center = (min_bound + max_bound) * 0.5
        min_bound = center - max_size / 2.
        max_bound = center + max_size / 2.
    else:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        min_bound = xyz_min
        max_bound = xyz_max

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    # xyz_range = np.linspace(min_bound, max_bound, num=np.max(reso))  # np.max(reso)
    xyz_range = [np.linspace(min_bound[i], max_bound[i], num=reso[i]) for i in range(3)]
    # xyz_range[:, 1], xyz_range[:, 2] = xyz_range[:, 2], xyz_range[:, 1]
    # query_points is a [32,32,32,3] array ..
    mesh_grid = np.meshgrid(*xyz_range)
    query_points = np.stack(mesh_grid, axis=-1).astype(np.float32)

    # signed distance is a [32,32,32] array
    # signed_distance = scene.compute_signed_distance(query_points)
    print('voxelizing given mesh: %s ...' % mesh_path)
    start = time.time()
    occupancy = scene.compute_occupancy(query_points).numpy()
    print('voxelizing done, time cost %.3f seconds' % (time.time() - start))
    #     np.save(voxel_grid_path, occupancy)
    # else:
    #     occupancy = np.load(voxel_grid_path)
    # print(occupancy.shape)
    xx, yy, zz = np.nonzero(occupancy)
    # xx, yy, zz = zz, xx, yy
    # xx, zz = reso[0] - zz, xx
    xx, yy, zz = yy, reso[0] - xx - 1, zz
    print(np.max(xx), np.max(yy), np.max(zz))
    return xx, yy, zz


def voxelize_mesh(mesh_path, reso, ratio):
    round_ratio = round(ratio, 1)
    voxel_grid_path = mesh_path.replace('.obj', '_voxel_grid_%s_%.1f.ply' % ('-'.join([str(i) for i in reso]), round_ratio))
    if not os.path.exists(voxel_grid_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        bbox = mesh.get_axis_aligned_bounding_box()
        scale = float(1./(np.max(bbox.max_bound - bbox.min_bound)))
        center = bbox.get_center()
        mesh = mesh.translate(-center)
        mesh.scale(scale, center=np.zeros([3, 1], dtype=np.float64))

        # Create a voxel grid from the point cloud with a voxel_size of 0.01
        print('voxelizing given mesh: %s ...' % mesh_path)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1./np.min(reso)/round_ratio)
        o3d.io.write_voxel_grid(voxel_grid_path, voxel_grid)
        print('voxelizing done.')
    else:
        voxel_grid = o3d.io.read_voxel_grid(voxel_grid_path)
        print('restore pre-saved voxel grid')

    voxels_all = voxel_grid.get_voxels()
    voxels_ind = np.array([i.grid_index for i in voxels_all])
    reso_center = np.array(reso) // 2
    voxel_grid_center = (np.max(voxels_ind, axis=0) + np.min(voxels_ind, axis=0)) // 2
    voxels_ind = voxels_ind + (reso_center - voxel_grid_center)
    # voxels_ind[:, 0] = reso[0] - voxels_ind[:, 0]
    # voxels_ind[:, 2] = reso[2] - voxels_ind[:, 2]

    # # Initialize a visualizer object
    # vis = o3d.visualization.Visualizer()
    # mesh_cor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #
    # # Create a window, name it and scale it
    # vis.create_window(window_name='Bunny Visualize', width=800, height=600)
    #
    # # Add the voxel grid to the visualizer
    # vis.add_geometry(mesh)
    # vis.add_geometry(mesh_cor)
    #
    # # We run the visualizater
    # vis.run()
    # # Once the visualizer is closed destroy the window and clean up
    # vis.destroy_window()

    xx = voxels_ind[:, 0]
    yy = voxels_ind[:, 1]
    zz = voxels_ind[:, 2]
    return xx, yy, zz

# voxelize_mes_2('../data/sdf/bunny.obj', [128, 128, 128], 0.8)

''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

@torch.no_grad()
def get_diffusion_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    # print('get_diffusion_rays: start', rgb_tr.shape)
    text_dir = rgb_tr
    # assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(train_poses) == len(Ks)
    H, W = HW
    # K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(train_poses), H, W, 3], device=text_dir.device)
    rays_d_tr = torch.zeros([len(train_poses), H, W, 3], device=text_dir.device)
    viewdirs_tr = torch.zeros([len(train_poses), H, W, 3], device=text_dir.device)
    imsz = [1] * len(text_dir)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=Ks[i], c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(text_dir.device))
        rays_d_tr[i].copy_(rays_d.to(text_dir.device))
        viewdirs_tr[i].copy_(viewdirs.to(text_dir.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    # print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return text_dir, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

