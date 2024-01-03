import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from . import grid
from .dvgo import DirectVoxGO
from .dmtet import DMTetGeometry
import nvdiffrast.torch as dr
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency
from .render import mesh
_FG_LUT = None

class DefSDF(torch.nn.Module):
    def __init__(self, init_sdf, init_deformation, grid_res):
        super(DefSDF, self).__init__()
        self.sdf = nn.Parameter(init_sdf)
        self.deformation = nn.Parameter(init_deformation)
        self.grid_res = grid_res
    
    def activate_deformation(self, grid_res):
        deformation = 1.0 / (grid_res * 2.) * torch.tanh(self.deformation)
        # deformation = torch.tanh(self.deformation) / grid_res
        return deformation
    
    def forward(self, grid_res):
        deformation = self.activate_deformation(grid_res)
        sdf = self.sdf
        return deformation, sdf
        
    

class DvgoDmtet(torch.nn.Module):
    def __init__(self, pretrained_model: DirectVoxGO, grid_res, dmtet_scale=torch.tensor([2., 2, 2]),
                 world_size=None, xyz_min=None, xyz_max=None,
                 rgbnet_dim=0, rgbnet_width=128, rgbnet_depth=3, **kwargs):
        super(DvgoDmtet, self).__init__()
        self.rgbnet_depth = rgbnet_depth
        self.rgbnet_dim = rgbnet_dim
        self.rgbnet_width = rgbnet_width
        if pretrained_model is not None:
            self.xyz_min = pretrained_model.xyz_min
            self.xyz_max = pretrained_model.xyz_max
            self.dmtet_scale = torch.maximum(self.xyz_max.abs(), self.xyz_min.abs()) * 2.
            # self.dmtet_scale = torch.maximum(self.xyz_max.abs().amax(), self.xyz_min.abs().amax()) * 2

            print('stage 2, bbox range ', self.xyz_min, self.xyz_max, self.dmtet_scale)
            # self.dmtet_scale = (self.xyz_max - self.xyz_min).amax()
            self.world_size = pretrained_model.world_size
            # background model
            self.background = pretrained_model.background
            # color field
            pretrained_model.k0.grid = nn.Parameter(
                F.avg_pool3d(pretrained_model.k0.get_dense_grid(), kernel_size=3, padding=1, stride=1))
            self.k0 = pretrained_model.k0
            self.rgbnet = pretrained_model.rgbnet
        else:
            self.xyz_min = xyz_min
            self.xyz_max = xyz_max
            self.dmtet_scale = dmtet_scale
            self.world_size = world_size
            self.background = grid.MLPBackground(4, 128, 3)

            k0_dim = 3
            if rgbnet_dim > 0:
                self.rgbnet = nn.Sequential(
                    nn.Linear(rgbnet_dim, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth - 2)
                    ],
                    nn.Linear(rgbnet_width, 3)
                )
                k0_dim = rgbnet_dim
            else:
                self.rgbnet = None
            self.k0 = grid.create_grid(
                'DenseGrid', channels=k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                init_method='uniform', init_mean=0.5, init_std=0.5)
        self.grid_res = grid_res
        # Geometry class for DMTet
        self.dmtet_geometry = DMTetGeometry(grid_res=grid_res, scale=self.dmtet_scale)

        # init with a ball, radius 0.3
        # init_sdf = 0.3 - torch.sqrt(torch.sum(self.dmtet_geometry.verts**2, dim=-1))
        # init with pretrained density
        self.density_bias = 0.5
        self.flip = torch.tensor([1., -1, 1])
        if pretrained_model is not None:
            # pretrained_model.density.grid = nn.Parameter(F.max_pool3d(pretrained_model.density.get_dense_grid(), kernel_size=3, padding=1, stride=1))
            init_sdf = pretrained_model.density(self.dmtet_geometry.verts * self.flip) - self.density_bias
        else:
            # init with a ball, radius 0.3
            init_sdf = 0.3 - torch.sqrt(torch.sum(self.dmtet_geometry.verts ** 2, dim=-1))

        init_deformation = torch.zeros([len(self.dmtet_geometry.verts), 3])
        self.defsdf = DefSDF(init_sdf=init_sdf, init_deformation=init_deformation, grid_res=grid_res)

        # render
        self.ctx = dr.RasterizeCudaContext() # changed opengl to cuda rendering, 
        
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        # dmtet_camera = PerspectiveCamera(fovyangle)
        focal = np.tan(fovyangle / 180.0 * np.pi * 0.5)
        self.proj_mtx = torch.tensor(projection(f=focal, far=1000.0, n=1.0, near_plane=0.1)).unsqueeze(dim=0)

        # self.proj_mtx = torch.from_numpy(projection(f=1., far=1000.0, n=1.0, near_plane=0.1)).unsqueeze(
        #     dim=0)
    

    def forward(self, ks, cam_pose, viewdirs, H, W, light_d, shading, ambient_ratio, **render_kwargs ):
        deformation, sdf = self.defsdf(self.grid_res)
        
        verts, faces = self.dmtet_geometry.get_mesh(deformation + self.dmtet_geometry.verts, sdf, with_uv=False)

        if 'filter' in render_kwargs:
            # # filter mesh
            imesh = mesh.Mesh(verts, faces)
            outlier_n_faces_threshold = 0.1
            imesh = mesh.remove_outlier(imesh, outlier_n_faces_threshold)
            verts, faces = imesh.v_pos, imesh.t_pos_idx

        m = Meshes(verts.unsqueeze(dim=0), faces.unsqueeze(dim=0))
        loss_smooth = mesh_normal_consistency(m)
        # m.verts_normals_padded()
        # Render the mesh into 2D image (get 3d position of each image plane)
        focals = ks[:, 0, 0] / H  # * 2.
        tex_pos, antilias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal = self.render_mesh(
            verts.unsqueeze(dim=0),
            faces.int(),
            cam_pose,
            focals,
            resolution=[H, W],
        )
        bg_type = render_kwargs['bg_type']
        # img_rgb = self.sample_rgb(viewdirs, tex_pos, hard_mask, bg_type)
        img_rgb = self.k0(tex_pos * self.flip)
        if self.rgbnet is not None:
            img_rgb = self.rgbnet(img_rgb)
        img_rgb = torch.sigmoid(img_rgb)

        # if shading == 'textureless':
        #     bg_colors = 0.
        # else:
        if bg_type < 0:
            bg_colors = self.background(viewdirs).reshape(img_rgb.shape)
        else:
            bg_colors = max(min(bg_type, 1.), 0.)

        if shading == 'normal':
            img_rgb = normal
        else:
            if shading != 'albedo':
                # lambertian shading
                light_d = light_d.reshape([-1, 1, 1, 3])
                l = safe_normalize(light_d - tex_pos) # [N, 1, 1, 3]
                lambertian = ambient_ratio + (1 - ambient_ratio) * torch.einsum('bhwc,bhwc->bhw', [normal, l]).clamp(min=0).unsqueeze(-1) # [N, 1]
                if shading == 'textureless':
                    img_rgb = lambertian.repeat([1, 1, 1, 3])
                elif shading == 'normal_light':
                    img_rgb = normal * lambertian
                else:  # 'lambertian'
                    img_rgb = img_rgb * lambertian

            # img_rgb = dr.antialias(img_rgb.contiguous(), rast, v_pos_clip, faces.int())  # [-1, H, W, 3]
            img_rgb = img_rgb * antilias_mask + bg_colors * (1 - antilias_mask)

        # img = img_feat.permute(0, 3, 1, 2)
        render_result = {
            'rgb_marched': img_rgb,
            'depth': depth,
            'normal_marched': normal,
            'loss_smooth': loss_smooth,
        }
        return render_result

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            camera_focal_b,
            resolution=[256, 256],
            spp=1,
    ):
        # camera_mv_bx4x4, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(len(camera_mv_bx4x4))
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=self.device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # Projection in the camera
        proj_mtx = self.proj_mtx.repeat(len(camera_focal_b), 1, 1)
        proj_mtx[:, 0, :] *= camera_focal_b.reshape([-1, 1])*resolution[0]/resolution[1]
        proj_mtx[:, 1, :] *= camera_focal_b.reshape([-1, 1])
        v_pos_clip = torch.matmul(v_pos, torch.transpose(proj_mtx, 1, 2))
        # normal
        v0 = v_pos[:, mesh_t_pos_idx_fx3[:, 0].long(), :3]
        v1 = v_pos[:, mesh_t_pos_idx_fx3[:, 1].long(), :3]
        v2 = v_pos[:, mesh_t_pos_idx_fx3[:, 2].long(), :3]
        faces_normals = torch.cross(v1 - v0, v2 - v0)

        verts_normals = torch.zeros_like(v_pos[:, :, :3])
        verts_normals = verts_normals.index_add(
            1, mesh_t_pos_idx_fx3[:, 0], faces_normals
        )
        verts_normals = verts_normals.index_add(
            1, mesh_t_pos_idx_fx3[:, 1], faces_normals
        )
        verts_normals = verts_normals.index_add(
            1, mesh_t_pos_idx_fx3[:, 2], faces_normals
        )
        verts_normals = safe_normalize(verts_normals)
        # face_normal_indices = (
        #     torch.arange(0, face_normals.shape[1], dtype=torch.int32, device='cuda')[:, None]).repeat(1, 3)
        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_pos_bxnx3.repeat([len(v_pos), 1, 1]), v_pos], dim=-1)  # Concatenate the pos  compute the supervision
        
        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution[0] * spp, resolution[1] * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
                gb_normal, _ = interpolate(verts_normals, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(hard_mask.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        # antialias_mask = dr.antialias(hard_mask.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        normal = gb_normal
        normal = dr.antialias(normal, rast, v_pos_clip, mesh_t_pos_idx_fx3).clamp(-1, 1)  # [-1, H, W, 3]

        depth = gb_feat[..., -2:-1].abs()
        ori_mesh_feature = gb_feat[..., :-4]
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal

    def sample_rgb(self, viewdirs, rgb_pos, hard_mask, bg_type):

        # tex_hard_mask = hard_mask
        # tex_pos = torch.cat([rgb_pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2)
        # tex_hard_mask = torch.cat(
        #         [tex_hard_mask[run_n_view + i_view: run_n_view + i_view + 1]
        #          for i_view in range(run_n_view)], dim=2)
        sampled_rgb = self.k0(rgb_pos * self.flip)
        sampled_rgb = torch.sigmoid(sampled_rgb)
        bg_rgb = self.background(viewdirs).reshape(sampled_rgb.shape)
        img_rgb = sampled_rgb * hard_mask + bg_rgb * (1 - hard_mask)
        return img_rgb

    def extract_3d_shape(self, save_path, texture_resolution=2048, **block_kwargs):
        '''
        Extract the 3D shape with texture map
        :param texture_resolution: the resolution for texure map
        :param block_kwargs:
        :return:
        '''

        # Step 1: predict geometry first
        deformation, sdf = self.defsdf(self.grid_res)
        
        mesh_v, mesh_f = self.dmtet_geometry.get_mesh(deformation + self.dmtet_geometry.verts, sdf, with_uv=False)
        # if 'filter' in render_kwargs:
        # # filter mesh
        imesh = mesh.Mesh(mesh_v, mesh_f)
        outlier_n_faces_threshold = 0.1
        imesh = mesh.remove_outlier(imesh, outlier_n_faces_threshold)
        mesh_v, mesh_f = imesh.v_pos, imesh.t_pos_idx
        # Step 2: use x-atlas to get uv mapping for the mesh
        from .mesh_utils import xatlas_uvmap, savemeshtes2
        uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(self.ctx, mesh_v, mesh_f, resolution=texture_resolution)

        # Step 3: Query the texture field to get the RGB color for texture map
        tex_feat = self.k0(gb_pos * self.flip)
        if self.rgbnet is not None:
            tex_feat = self.rgbnet(tex_feat)
        tex_feat = torch.sigmoid(tex_feat)
        background_feature = torch.zeros_like(tex_feat)
        # Merge them together
        img_feat = tex_feat * mask.float() + background_feature * (1 - mask.float())

        savemeshtes2(mesh_v.data.cpu().numpy(),
                     uvs.data.cpu().numpy(),
                     mesh_f.data.cpu().numpy(),
                     mesh_tex_idx.data.cpu().numpy(),
                     img_feat,
                     save_path)

    def init_img(self, *args):
        pass

    def get_kwargs(self):
        return {
            'pretrained_model': None,
            'grid_res': self.grid_res,
            'dmtet_scale': self.dmtet_scale,
            'world_size': self.world_size,
            'xyz_min': self.xyz_min,
            'xyz_max': self.xyz_max,
            'rgbnet_dim': self.rgbnet_dim,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth,
        }


def projection(f=0.1, n=1.0, far=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / f, 0, 0, 0],
         [0, n / -f, 0, 0],
         [0, 0, -(far + near_plane) / (far - near_plane), -(2 * far * near_plane) / (far - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)