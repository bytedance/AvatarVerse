import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
try:
    render_utils_cuda = load(
            name='render_utils_cuda',
            sources=[
                os.path.join(parent_dir, path)
                for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
            verbose=True)

    total_variation_cuda = load(
            name='total_variation_cuda',
            sources=[
                os.path.join(parent_dir, path)
                for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
            verbose=True)
except:
    render_utils_cuda = total_variation_cuda = None
    pass

def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    elif type == 'TriPlane':
        return TriPlane(**kwargs)
    else:
        raise NotImplementedError

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, init_method='normal', init_mean=0., init_std=1., 
                 add_blob=False, add_outward=False, alpha=False, bias=False, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.add_blob = add_blob
        self.add_outward = add_outward
        self.register_buffer('xyz_min', torch.Tensor(xyz_min.clone()))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max.clone()))

        if bias:
            self.bias = DenseGrid(channels=channels, world_size=[300, 300, 300], xyz_min=xyz_min, xyz_max=xyz_max,
                                  init_method='constant', init_mean=0., init_std=0.5,
                                  add_blob=False, add_outward=False, alpha=False, bias=False)
            self.bias.grid.requires_grad = False
        else:
            self.bias = None
        self.blob_scale = nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=False)
        self.blob_center = nn.Parameter(torch.tensor([0., 0., 0.]), requires_grad=False)
        # self.bias = None
        assert init_method in ['normal', 'uniform', 'constant']
        if init_method == 'normal':
            print('dvgo: init voxel with normal distribution')
            self.grid = nn.Parameter(torch.randn([1, channels, *world_size]) * init_std + init_mean)
        elif init_method == 'uniform':
            print('dvgo: init voxel with uniform distribution')
            self.grid = nn.Parameter(torch.rand([1, channels, *world_size]) + init_mean - 0.5)
        elif init_method == 'constant':
            print('dvgo: init voxel with constant distribution')
            self.grid = nn.Parameter(torch.ones([1, channels, *world_size]) * init_mean)
        else:
            raise 'not implemrnt init method: \'%s\''% init_method

        if alpha:
            self.alpha = nn.Parameter(torch.tensor([1.]))
        else:
            self.alpha = torch.tensor([1.])

        self.init_smooth_conv(ksize=5, sigma=0.8)

    def init_smooth_conv(self, ksize=3, sigma=1.):
        self.smooth_sdf = ksize > 0
        if ksize > 0:
            self.smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- " * 10 + "init smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -" * 10)

    def _gaussian_3dconv(self, ksize=3, sigma=1.):
        x = np.arange(-(ksize//2),ksize//2 + 1,1)
        y = np.arange(-(ksize//2),ksize//2 + 1,1)
        z = np.arange(-(ksize//2),ksize//2 + 1,1)
        xx, yy, zz = np.meshgrid(x,y,z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
        kernel = torch.from_numpy(kernel).to(self.grid)
        m = nn.Conv3d(1,1,ksize,stride=1,padding=ksize//2, padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum()
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def set_blob_para(self, scale=torch.tensor([1., 1., 1.]), center=torch.tensor([0., 0., 0.])):
        self.blob_scale = nn.Parameter(scale.to(self.blob_scale.device),  requires_grad=False)
        self.blob_center = nn.Parameter(center.to(self.blob_center.device),  requires_grad=False)

    def density_blob(self, x, mean=5., radius=0.5):
        d = ((x*self.blob_scale - self.blob_center) ** 2).sum(-1)
        # g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))
        g = mean * (1 - torch.sqrt(d) / radius)
        # g[g < -10] = -10.

        # # extra_d = mean * torch.tensor(torch.abs(x-torch.tensor([0, 0, 0.4])).amax(dim=-1) < 0.1, dtype=torch.int)
        return g
    
    def normal_outward(self, x):
        
        return safe_normalize(x)

    def forward(self, xyz, smooth=False):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        # kernal_size = self.world_size.min().item() // 32 * 2 + 1
        # if kernal_size > 1:
        #     self_grid = F.avg_pool3d(self.grid, kernel_size=kernal_size, padding=1, stride=1)
        #     out = F.grid_sample(self_grid, ind_norm, mode='bilinear', align_corners=True)
        # else:
        if smooth:
            sdf_grid = self.smooth_conv(self.grid)
            out = F.grid_sample(sdf_grid, ind_norm, mode='bilinear', align_corners=True)
        else:
            out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)

        if self.add_outward:
            out = out + self.normal_outward(xyz).reshape(out.shape)
        bias = None
        if self.bias:
            bias = self.bias(xyz).reshape(out.shape)
            out = out + bias
        if self.add_blob:
            blob = self.density_blob(xyz).reshape(out.shape)
            if bias is not None:
                blob = torch.where(bias < 0.1, blob, torch.zeros_like(blob))
            out = out + blob
        out = torch.maximum(self.alpha, torch.tensor([1.])).to(out.device) * out
        # # hacked code for hand persistency
        # if self.bias and self.channels == 1:
        #     hand_mask = torch.logical_and(torch.abs(xyz[..., 0]) > 0.3, xyz[..., 1] < -0.1)
        #     hand_mask = hand_mask.reshape(out.shape)
        #     # print(hand_mask.shape, out.shape, bias.shape)

        #     # hand_mask = torch.logical_and(hand_mask, bias > 1.)
        #     out = torch.where(hand_mask, bias+blob, out)

        #     lower_foot_mask = xyz[..., 1] > 0.55
        #     lower_foot_mask = lower_foot_mask.reshape(hand_mask.shape)
        #     out = torch.where(lower_foot_mask, bias+blob, out)
        #     # print(torch.sum(hand_mask))
        #     # print(out.shape)
        return out
    
    def crop_and_scale_volume_grid(self, crop_xyz_min, crop_xyz_max, new_world_size):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, new_world_size[0]),
            torch.linspace(0, 1, new_world_size[1]),
            torch.linspace(0, 1, new_world_size[2]),
        ), -1)
        crop_xyz = crop_xyz_min * (1-interp) + crop_xyz_max * interp
        shape = crop_xyz.shape[:-1]
        crop_xyz = crop_xyz.reshape(1,1,1,-1,3)
        ind_norm = ((crop_xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        crop_grid = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        crop_grid = crop_grid.reshape(1, self.channels, *shape)

        # ind_start = ((crop_xyz_min - self.xyz_min) / (self.xyz_max - self.xyz_min) * self.world_size).flip((-1,)).long()
        # ind_end = ((crop_xyz_max - self.xyz_min) / (self.xyz_max - self.xyz_min) * self.world_size).flip((-1,)).long()
        # # print(crop_xyz_min, self.xyz_min, crop_xyz_max, self.xyz_max)
        # # print(ind_start)
        # # print(ind_end)
        # interp_grid = F.interpolate(self.grid.data, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        # crop_grid = interp_grid[:, :, ind_start[0]:ind_end[0], ind_start[1]:ind_end[1], ind_start[2]:ind_end[2]]
        self.grid = nn.Parameter(crop_grid)
        self.xyz_min = crop_xyz_min.clone()
        self.xyz_max = crop_xyz_max.clone()
        self.world_size = new_world_size.clone()
        self.scale_volume_grid(new_world_size)

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size.clone()
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}'


class MLPBackground(nn.Module):
    def __init__(self, viewbase_pe=4, hidden_width=128, layers=3, out_dim=3):
        super(MLPBackground, self).__init__()
        self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
        dim0 = 3 + 3 * viewbase_pe * 2

        self.net = nn.Sequential(
            nn.Linear(dim0, hidden_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.ReLU(inplace=True))
                for _ in range(layers - 2)
            ],
            nn.Linear(hidden_width, out_dim)
        )
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, ray_dirs):
        '''
        xyz: NX3 global coordinates to query
        '''
        ray_pts_emb = (ray_dirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        ray_pts_emb = torch.cat([ray_dirs, ray_pts_emb.sin(), ray_pts_emb.cos()], -1)
        rgb = self.net(ray_pts_emb).squeeze(-1)
        rgb = torch.sigmoid(rgb)
        return rgb

''' Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
'''
class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R+R+Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[...,[0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape,self.channels)
        else:
            out = compute_tensorf_val(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X,1], mode='bilinear', align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y,1], mode='bilinear', align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z,1], mode='bilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.x_vec[:,:,1:], self.x_vec[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.y_vec[:,:,1:], self.y_vec[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.z_vec[:,:,1:], self.z_vec[:,:,:-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([
                torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0,:,:,0]),
                torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0,:,:,0]),
                torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0,:,:,0]),
            ])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0,:,:,0]) + \
                   torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0,:,:,0]) + \
                   torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0,:,:,0])
            grid = grid[None,None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}, n_comp={self.config["n_comp"]}'

def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = torch.cat([
        xy_feat * z_feat,
        xz_feat * y_feat,
        yz_feat * x_feat,
    ], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat

def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, mask=None, xyz_min=None, xyz_max=None, bound_radius=None):
        super(MaskGrid, self).__init__()
        # if path is not None:
        #     st = torch.load(path)
        #     self.act_softplus = True
        #     self.mask_cache_thres = mask_cache_thres
        #     density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
        #     if self.act_softplus:
        #         alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
        #     else:
        #         alpha = 1.- torch.exp(-torch.nn.ReLU()(density) * st['model_kwargs']['voxel_size_ratio'])
        #     mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
        #     xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
        #     xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        # else:
        mask = mask.bool()
        xyz_min = torch.Tensor(xyz_min.clone())
        xyz_max = torch.Tensor(xyz_max.clone())
        
        self.bound_mask = torch.ones_like(mask).to(mask.dtype).to(mask.device)
        if bound_radius is not None:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(xyz_min[0], xyz_max[0], mask.shape[0]),
                torch.linspace(xyz_min[1], xyz_max[1], mask.shape[1]),
                torch.linspace(xyz_min[2], xyz_max[2], mask.shape[2]),
            ), -1)
            bbox_scale = 1 + 1./torch.Tensor(list(mask.shape))
            d = torch.sum((self_grid_xyz * bbox_scale) ** 2, dim=-1)
            self.bound_mask[d>bound_radius] = False

        self.register_buffer('mask', mask)
        self.mask = mask
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask&self.bound_mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'


def grid_sample(input, grid):
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)


# ----------------------------------------------------------------------------
enabled = True  # Enable the custom op by setting this to true.
def _should_use_custom_op():
    return enabled


# ----------------------------------------------------------------------------

class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid


# ----------------------------------------------------------------------------

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid  # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)

        assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid

# ----------------------------------------------------------------------------
class TriPlane(nn.Module):
    def __init__(self, channels: int, world_size: int, xyz_min, xyz_max, add_blob=False, alpha=False, **kwargs):
        super(TriPlane, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.add_blob = add_blob
        self.register_buffer('xyz_min', torch.Tensor(xyz_min.clone()))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max.clone()))

        self.img_feat_dim = 32
        self.img_feat_w = 128
        self.tri_planes = nn.Parameter(torch.ones(3, 32, self.img_feat_w, self.img_feat_w))
        self.spatial_conv = nn.Conv2d(self.img_feat_dim, self.img_feat_dim, 3, padding=(1, 1))
        nn.init.kaiming_uniform_(self.tri_planes, a=np.sqrt(5))
        self.fc_net = nn.Sequential(
            nn.Linear(self.img_feat_dim, 16), nn.ReLU(inplace=True),
            # nn.Linear(32, 32), nn.ReLU(inplace=True),
            # nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
            nn.Linear(16, self.channels),
        )
        nn.init.constant_(self.fc_net[-1].bias, 0)
        if alpha:
            self.alpha = nn.Parameter(torch.tensor([1.]))
        else:
            self.alpha = torch.tensor([1.])
        pass

    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, -1, 3)
        # triplanes = self.spatial_conv(self.tri_planes)
        triplanes = self.tri_planes
        normalized_tex_pos = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        normalized_tex_pos = torch.clamp(normalized_tex_pos, 0, 1)
        normalized_tex_pos = normalized_tex_pos * 2.0 - 1.0
        x_feat = grid_sample(triplanes[0:1],
                             torch.cat([normalized_tex_pos[..., 0:1], normalized_tex_pos[..., 1:2]], dim=-1).detach())
        y_feat = grid_sample(triplanes[1:2],
                             torch.cat([normalized_tex_pos[..., 1:2], normalized_tex_pos[..., 2:3]], dim=-1).detach())
        z_feat = grid_sample(triplanes[2:3],
                             torch.cat([normalized_tex_pos[..., 0:1], normalized_tex_pos[..., 2:3]], dim=-1).detach())

        final_feat = (x_feat + y_feat + z_feat)
        final_feat = final_feat.reshape(self.img_feat_dim, -1).T.reshape(*shape, self.img_feat_dim)
        out = self.fc_net(final_feat)
        if self.channels == 1:
            out = out.squeeze(-1)
        if self.add_blob:
            out = out + self.density_blob(xyz).reshape(out.shape)

        out = self.alpha * out
        return out

    @staticmethod
    def density_blob(x, mean=8., radius=0.3):
        scale = torch.tensor([1, 1., 1])
        d = ((x * scale) ** 2).sum(-1)
        # g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))
        g = mean * (1 - torch.sqrt(d) / radius)

        # # extra_d = mean * torch.tensor(torch.abs(x-torch.tensor([0, 0, 0.4])).amax(dim=-1) < 0.1, dtype=torch.int)
        return g

    def crop_and_scale_volume_grid(self, crop_xyz_min, crop_xyz_max, new_world_size):
        scaled_crop_xyz_min = (crop_xyz_min-self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1.
        scaled_crop_xyz_max = (crop_xyz_max-self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1.
        interp_x = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, new_world_size[0]),
            torch.linspace(0, 1, new_world_size[1]),
        ), -1).unsqueeze(0)
        crop_x = scaled_crop_xyz_min[[0, 1]] * (1 - interp_x) + scaled_crop_xyz_max[[0, 1]] * interp_x
        x_feat = grid_sample(self.tri_planes[0:1], crop_x)

        interp_y = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, new_world_size[1]),
            torch.linspace(0, 1, new_world_size[2]),
        ), -1).unsqueeze(0)
        crop_y = scaled_crop_xyz_min[[1, 2]] * (1 - interp_y) + scaled_crop_xyz_max[[1, 2]] * interp_y
        y_feat = grid_sample(self.tri_planes[1:2], crop_y)

        interp_z = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, new_world_size[0]),
            torch.linspace(0, 1, new_world_size[2]),
        ), -1).unsqueeze(0)
        crop_z = scaled_crop_xyz_min[[0, 2]] * (1 - interp_z) + scaled_crop_xyz_max[[0, 2]] * interp_z
        z_feat = grid_sample(self.tri_planes[2:3], crop_z)
        crop_feat = torch.cat([x_feat, y_feat, z_feat], dim=0)

        self.tri_planes = nn.Parameter(crop_feat)
        self.xyz_min = crop_xyz_min.clone()
        self.xyz_max = crop_xyz_max.clone()
        self.world_size = new_world_size.clone()
        self.scale_volume_grid(new_world_size)

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size.clone()
        if self.channels == 0:
            self.tri_planes = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.tri_planes = nn.Parameter(
                F.interpolate(self.tri_planes.data, size=(self.img_feat_w, self.img_feat_w), mode='bilinear', align_corners=True))

    def get_dense_grid(self):
        raise 'not supported for tri plane'
        return self.grid
