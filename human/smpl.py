import sys
import os
import trimesh
import matplotlib
import pickle
import torch
from PIL import Image
from scipy.io import loadmat
import numpy as np
from human.pose_loader import GradLayer
from human.poses import *

try:
    import kaolin as kal
except:
    kal = None
import smplx


class DensePoseSMPL:
    def __init__(self, src_path='./human/smpl/', crop_mode='half', pose_type='nerf'):
        assert crop_mode in ['head', 'shoulder', 'full']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pose_type = pose_type
        self.crop_mode = crop_mode

        data_filename = os.path.join(src_path, "DensePose/UV_Processed.mat")
        # rename your .pkl file or change this string
        verts_filename = os.path.join(src_path, "DensePose/smpl_model.pkl")
        self.smpl = smplx.create(verts_filename).to(self.device)

        pose = GENERAL_STANDING.to(self.device)
        self.focus = GENERAL_STANDING_FOCUS

        if crop_mode =='head':
            self.focus = GENERAL_NO_FOCUS
        elif crop_mode =='shoulder':
            self.focus = GENERAL_SHOULDER_FOCUS
        v_template = self.smpl_stand_pose_verts(pose)  # (6890, 3)
        avatar_size_scale = torch.tensor([1, 1., 1.]).to(self.device).float()
        v_template = v_template * avatar_size_scale

        # edited smpl mesh
        # mesh = trimesh.load(os.path.join(src_path, './hulk1.obj'))
        # v_template = torch.tensor(mesh.vertices).to(v_template)

        self.smpl_vertices = v_template.unsqueeze(0).to(self.device)
        self.smpl_faces = torch.tensor(self.smpl.faces.astype(np.int64), device=self.device)

        # tex_filename = os.path.join(src_path, "DensePose/texture_from_SURREAL.png")
        # with Image.open(tex_filename) as image:
        #     np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        # tex = torch.from_numpy(np_image / 255.)[None].to(device)

        ALP_UV = loadmat(data_filename)
        self.verts_ind = torch.from_numpy((ALP_UV["All_vertices"]).astype(int)).squeeze().to(self.device) - 1  # (7829,)
        # U = torch.Tensor(ALP_UV['All_U_norm']).to(device)  # (7829, 1)
        # V = torch.Tensor(ALP_UV['All_V_norm']).to(device)  # (7829, 1)
        self.faces_org = torch.from_numpy((ALP_UV['All_Faces'] - 1).astype(int)).to(self.device)  # (13774, 3)
        self.faces = self.faces_org.clone()
        self.vertices = v_template[self.verts_ind].unsqueeze(0).to(self.device)  # (1, 7829, 3)
        self.face_indices = torch.Tensor(ALP_UV['All_FaceIndices']).squeeze().long().to(self.device)  # (13774,)
        cmap = matplotlib.cm.get_cmap('viridis', 25)
        self.cmap = torch.tensor((np.array(cmap.colors)[:, :3] * 255)).to(self.device).float()
        self.face_colors = self.cmap[self.face_indices]  # (13774, 3)

        self.subset = torch.ones_like(self.verts_ind).to(torch.bool)
        self.crop_pose(crop_mode)
        # normalize
        self.normalize_center = (torch.amax(self.vertices[:, self.subset], dim=-2, keepdim=True) +
                            torch.amin(self.vertices[:, self.subset], dim=-2, keepdim=True)) / 2.
        self.normalize_center[:, 0, 0] = 0.
        self.normalize_center[:, 0, 2] = 0.
        self.vertex_size = torch.max(torch.amax(self.vertices[:, self.subset], dim=-2) -
                                torch.amin(self.vertices[:, self.subset], dim=-2))
        self.vertices = (self.vertices - self.normalize_center) / self.vertex_size
        self.smpl_vertices = (self.smpl_vertices - self.normalize_center) / self.vertex_size

        self.cvt_mask = torch.tensor([[[1., -1, 1, 1],
                                  [1, 1, -1, -1],
                                  [1, -1, 1, 1],
                                  [1, 1, 1, 1]]]).float().to(self.device)
        self.sobel_op = GradLayer(self.device)

        self.nb_faces = len(self.faces)
        self.face_attributes = [
            torch.cat([self.face_colors.reshape([1, -1, 1, 3])] * 3, dim=2),
            torch.ones((1, self.nb_faces, 3, 1), device=self.device)
        ]

    def get_mesh(self):
        # return self.vertices.cpu().numpy()[0], self.faces_org.cpu().numpy()
        return self.smpl_vertices.cpu().numpy()[0], self.smpl_faces.cpu().numpy()

    def smpl_stand_pose_verts(self, pose):

        # Convert to rotation matrix
        v_template = self.smpl(body_pose=pose).vertices.squeeze(0).detach().to(self.device)
        return v_template

    def set_new_pose(self, pose):
        v_template = self.smpl(body_pose=pose).vertices.squeeze(0).detach().to(self.device)
        self.vertices = v_template[self.verts_ind].unsqueeze(0).to(self.device)  # (1, 7829, 3)
        self.vertices = (self.vertices - self.normalize_center) / self.vertex_size

    def crop_pose(self, crop_mode):
        min_height = torch.min(self.vertices[0][:, 1])
        human_height = torch.max(self.vertices[0][:, 1]) - min_height
        if crop_mode == 'head':
            crop_height = 0.85
        elif crop_mode == 'shoulder':
            crop_height = 0.7
        else:
            crop_height = 0
        self.subset = self.vertices[0][:, 1] > crop_height * human_height + min_height

        # process faces
        # face_mask = torch.sum(self.subset[self.faces.reshape(-1)].reshape([-1, 3]), dim=-1) == 3
        # self.faces = self.faces[face_mask]
        # self.face_indices = self.face_indices[face_mask]
        # self.face_colors = self.face_colors[face_mask]

    def show_3d(self):

        mesh = trimesh.Trimesh(vertices=self.vertices.cpu().numpy()[0],
                               faces=self.faces.cpu().numpy(),
                               face_colors=self.face_colors)
        mesh.show()

    def render(self, cam_pos, cam_k, HW, vis=False, out_type='depth', **kwargs):
        fov_angles = -torch.arctan(cam_k[:, 1, 2]*2 / cam_k[:, 1, 1] / 2).cpu().numpy() * 2.
        cam_projs = []
        for ang in fov_angles:
            cam_proj = kal.render.camera.generate_perspective_projection(ang, float(HW[1])/HW[0])
            cam_projs.append(cam_proj)
            # print(ang, cam_proj)

        cam_projs = torch.stack(cam_projs, dim=0).to(self.device)
        # cam_proj = torch.tensor([[1.5, 1.5, 1.], [1., 1., 1.]], device='cuda').float()

        if self.pose_type == 'nerf':
            # a dirty conversion for different coordinate system
            trans_cam_pos = cam_pos * self.cvt_mask

            trans_cam_pos[:, :, 0] = -trans_cam_pos[:, :, 0]
            trans_cam_pos[:, :3, :3] = trans_cam_pos[:, :3, :3].permute(0, 2, 1)
        else:
            trans_cam_pos = torch.eye(4).unsqueeze(0).repeat([len(cam_pos), 1, 1]).to('cuda')
            trans_cam_pos[:, :3, :3] = cam_pos[:, :3, :3]
            t = torch.linalg.inv(trans_cam_pos) @ cam_pos
            trans_cam_pos[:, :2, :] *= -1
            trans_cam_pos[:, :3, -1] = -t[:, :3, -1]

        rot = trans_cam_pos[..., :3, :3].to(self.device).float()
        trans = trans_cam_pos[..., :3, 3].to(self.device).float()
        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                self.vertices,
                self.faces, cam_projs,
                camera_rot=rot, camera_trans=trans
            )

        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd
        face_attributes = [torch.cat([i]*len(cam_pos)) for i in self.face_attributes]
        with_depth_feattures = face_attributes + [face_vertices_camera[:, :, :, -1:]] + [torch.cat([face_normals.unsqueeze(2)] * 3, dim=2)]
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            HW[0], HW[1], face_vertices_camera[:, :, :, -1],
            face_vertices_image, with_depth_feattures, face_normals[:, :, -1],
            rast_backend='cuda')

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        face_colors, mask, depth_map_org, normal_map = image_features
        # depth_map -= torch.amin(depth_map, dim=[1,2,3], keepdim=True)
        # depth_map /= torch.amax(depth_map, dim=[1,2,3], keepdim=True)
        depth_map_org = -depth_map_org
        depth_map_img = depth_map_org - 0.2
        depth_map_img /= 5.
        depth_map_img = torch.clamp(depth_map_img, 0., 1.)
        depth_map_img = depth_map_img.float().permute([0, 3, 1, 2])
        depth_map = torch.cat([depth_map_img] * 3, dim=1)

        x, y = self.sobel_op.sobel(depth_map_org.permute([0, 3, 1, 2]))
        z = torch.ones_like(x) * 0.001
        x[depth_map_img < 0.] = 0
        y[depth_map_img < 0.] = 0
        normal = torch.cat([y, x, z], axis=1)[:, [2, 1, 0]]
        normal = normal / torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
        normal_map = (normal * 127.5 + 127.5).clip(0., 255.).float() / 255.
        # normal_map = normal_map.float().permute([0, 3, 1, 2])

        # normal_map_mask = torch.sum(torch.abs(normal_map), dim=-1, keepdim=True) > 1e-3
        # normal_map[...,  1] *= -1
        # normal_map = (normal_map/2.+0.5) * normal_map_mask
        # normal_map = normal_map.float().permute([0, 3, 1, 2])
        face_colors = mask * face_colors + (1-mask) * self.cmap[0]
        face_colors = face_colors.float().permute([0, 3, 1, 2])
        if out_type == 'depth':
            return depth_map
        elif out_type == 'normal':
            return normal_map
        elif out_type == 'dense_pose':
            return face_colors/255.
        # image = torch.clamp(image * mask, 0., 1.)
        # image = torch.clamp(image, 0., 1.)
        return face_colors, mask, depth_map


if __name__ == '__main__':
    densepose = DensePoseSMPL('./smpl', crop_mode='full')
    densepose.show_3d()
