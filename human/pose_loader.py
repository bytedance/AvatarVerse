# imports
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import torch
try:
    import kaolin as kal
except:
    kal = None
from PIL import Image
from torch.nn import functional as F
import torch
# DATA_PATH = "./datasets/json/"

# load json
# task1+2_train.json is quite big. In case of memory out, load it by parts. See json_loader_part()
# id_list is to record id for each image in dataset to help finding correspondence between input and target. You need to
# write out id into your json during test in order to allow online evaluation
# load task1+2_train.json by parts in case the whole file is too big


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def json_loader_part(data_path):
    target_list = []
    data = json.load(open(data_path))
    length = len(data)
    start = min([int(i) for i in data.keys()])
    for i in range(start, start + length):
        sample_3d = np.zeros([1, 17, 3])
        for j in range(17):
            sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
            sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
            sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
        # sample_3d[6:] = sample_3d[5:-1]
        # sample_3d[5] = (sample_3d[6] + sample_3d[7]) / 2.
        target_list.append(sample_3d)
    return target_list


def get_limb(X, Y, Z=None, id1=0, id2=1):
    if Z is not None:
        return np.concatenate((np.expand_dims(X[id1], 0), np.expand_dims(X[id2], 0)), 0), \
            np.concatenate((np.expand_dims(Y[id1], 0), np.expand_dims(Y[id2], 0)), 0), \
            np.concatenate((np.expand_dims(Z[id1], 0), np.expand_dims(Z[id2], 0)), 0)
    else:
        return np.concatenate((np.expand_dims(X[id1], 0), np.expand_dims(X[id2], 0)), 0), \
            np.concatenate((np.expand_dims(Y[id1], 0), np.expand_dims(Y[id2], 0)), 0)


# draw wholebody skeleton
# conf: which joint to draw, conf=None draw all
def draw_skeleton(vec, conf=None, pointsize=None, figsize=None, plt_show=False, save_path=None, inverse_z=True,
                  fakebbox=True, background=None):
    _, keypoint, d = vec.shape
    if keypoint == 17:
        X = vec
        if (d == 3) or ((d == 2) and (background == None)):
            X = X - (X[:, 11:12, :] + X[:, 12:13, :]) / 2.0
        list_branch_head = [(0, 1), (1, 3), (0, 2), (2, 4)]

        list_branch_left_arm = [(5, 7), (7, 9), ]

        list_branch_right_arm = [(6, 8), (8, 10)]

        list_branch_body = [(5, 6), (6, 12), (11, 12), (5, 11)]
        list_branch_right_foot = [(12, 14), (14, 16)]
        list_branch_left_foot = [(11, 13), (13, 15)]
    else:
        print('Not implemented this skeleton')
        return 0

    if d == 3:
        fig = plt.figure()
        if figsize is not None:
            fig.set_size_inches(figsize, figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.elev = 10
        ax.grid(False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if inverse_z:
            zdata = -X[0, :, 1]
        else:
            zdata = X[0, :, 1]
        xdata = X[0, :, 0]
        ydata = X[0, :, 2]
        if conf is not None:
            xdata *= conf[0, :].numpy()
            ydata *= conf[0, :].numpy()
            zdata *= conf[0, :].numpy()
        if pointsize is None:
            ax.scatter(xdata, ydata, zdata, c='r')
        else:
            ax.scatter(xdata, ydata, zdata, s=pointsize, c='r')

        if fakebbox:
            max_range = np.array(
                [xdata.max() - xdata.min(), ydata.max() - ydata.min(), zdata.max() - zdata.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            if background is not None:
                WidthX = Xb[7] - Xb[0]
                WidthY = Yb[7] - Yb[0]
                WidthZ = Zb[7] - Zb[0]
                arr = np.array(background.getdata()).reshape(background.size[1], background.size[0], 3).astype(float)
                arr = arr / arr.max()
                stepX, stepZ = WidthX / arr.shape[1], WidthZ / arr.shape[0]

                X1 = np.arange(0, -Xb[0] + Xb[7], stepX)
                Z1 = np.arange(Zb[7], Zb[0], -stepZ)
                X1, Z1 = np.meshgrid(X1, Z1)
                Y1 = Z1 * 0.0 + Zb[7] + 0.01
                ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, facecolors=arr, shade=False)

        for (id1, id2) in list_branch_head:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='red')
        for (id1, id2) in list_branch_body:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='orange')
        for (id1, id2) in list_branch_left_arm:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='blue')
        for (id1, id2) in list_branch_right_arm:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='violet')
        for (id1, id2) in list_branch_left_foot:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
                ax.plot(x, y, z, color='pink')

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close()

    if d == 2:
        fig = plt.figure()
        if figsize is not None:
            fig.set_size_inches(figsize, figsize)
        ax = plt.axes()
        ax.axis('off')
        if background is not None:
            im = ax.imshow(background)
            ydata = X[0, :, 1]
        else:
            ydata = -X[0, :, 1]
        xdata = X[0, :, 0]
        if conf is not None:
            xdata *= conf[0, :].numpy()
            ydata *= conf[0, :].numpy()
        if pointsize is None:
            ax.scatter(xdata, ydata, c='r')
        else:
            ax.scatter(xdata, ydata, s=pointsize, c='r')

        for (id1, id2) in list_branch_head:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='red')
        for (id1, id2) in list_branch_body:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='orange')
        for (id1, id2) in list_branch_left_arm:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='blue')
        for (id1, id2) in list_branch_right_arm:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='violet')
        for (id1, id2) in list_branch_left_foot:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
                x, y = get_limb(xdata, ydata, None, id1, id2)
                ax.plot(x, y, color='pink')

        if fakebbox:
            max_range = np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
            for xb, yb in zip(Xb, Yb):
                ax.plot([xb], [yb], 'w')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close()


def cvt_human35_to_coco(pose):
    transfer_index = [0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    new_pose = pose[..., transfer_index, :]
    new_pose[..., 1, :] = (pose[..., 5, :] + pose[..., 6, :]) /2.
    return new_pose


def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)


            # print(colors[i])
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


class HumanPose:
    def __init__(self, human_pose_path='./human/3d_pose.npy', crop_mode='head', pose_type='nerf'):
        assert crop_mode in ['head', 'shoulder', 'half', 'full']
        human_pose = np.load(human_pose_path)
        human_pose[:, 2] = -human_pose[:, 2]
        self.subset = np.arange(18).reshape([1, 18])
        self.crop_pose(crop_mode)
        self.pose_type = pose_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #  normalize
        pose_mask = self.subset[0] >= 0
        center = 0.5*(np.max(human_pose[pose_mask, :], axis=0) + np.min(human_pose[pose_mask, :], axis=0))
        max_size = np.max(np.max(human_pose[pose_mask, :], axis=0) - np.min(human_pose[pose_mask, :], axis=0))
        human_pose = (human_pose - center) / max_size

        height_bias = np.array([0., 0., 0])
        human_pose += height_bias
        # 4x18
        self.human_pose = torch.cat([torch.tensor(human_pose).float(), torch.ones(len(human_pose), 1)], dim=-1).T.float().to(self.device)

    def render(self, cam_pose, cam_k, HW, vis=False, **kwargs):
        '''
        human_pose: 1x17x3,
        cam_pose: bx4x4,
        cam_k: bx3x3
        '''
        cur_camk = cam_k.clone()
        cur_camk[:, 0] *= (HW[1] / cur_camk[:, 0, 2, None] / 2.)
        cur_camk[:, 1] *= (HW[0] / cur_camk[:, 1, 2, None] / 2.)
        pose_in_cam = torch.matmul(torch.linalg.inv(cam_pose)[:, :3, :], self.human_pose)
        uvd = torch.matmul(cur_camk, pose_in_cam).permute([0, 2, 1])
        uv = uvd[:, :, :2]/(uvd[:, :, 2:] + 1e-8)
        bsz = uv.shape[0]
        pose_img = np.zeros([bsz, HW[0], HW[1], 3])
        for b in range(bsz):
            pose_img[b] = draw_bodypose(pose_img[b], uv[b].cpu().numpy(), self.subset)/255.
            if vis:
                plt.imshow(pose_img[b])
                plt.show()
                plt.savefig(f'./pose_{b}.jpg')
        pose_img = torch.tensor(pose_img).float().permute([0, 3, 1, 2])
        # pose_img = torch.flip(pose_img, (3,))
        return pose_img

    def crop_pose(self, mode):
        if mode == 'head':
            self.subset[:, [2,3,4,5,6,7,8,9,10,11,12,13]] = -1
        elif mode == 'shoulder':
            self.subset[:, [4, 7, 8, 9, 10, 11, 12, 13]] = -1
        elif mode == 'half':
            self.subset[:, [9, 10, 12, 13]] = -1
        else:
            pass


class GradLayer:
    def __init__(self, device):
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.weight_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(device)
        # self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        # self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def sobel(self, x):
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x_v = F.conv2d(x, self.weight_v, padding=1)

        # x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x_h, x_v


import os
from human.joint_orders import SMPL_INDEX_FOR_OPENPOSE18
from human.poses import *
import smplx
class HumanPoseSMPLX(HumanPose):
    def __init__(self, src_path='./human/smpl/', crop_mode='head', pose_type='nerf'):
        super(HumanPoseSMPLX, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        verts_filename = os.path.join(src_path, "SMPLX_NEUTRAL.npz")
        self.crop_mode = crop_mode
        self.smpl = smplx.create(verts_filename).to(self.device)

        pose = GENERAL_STANDING[:, :-2*3].to(self.device)
        # pose = IRONMAN_STANDING_FIRE.to(self.device)
        self.focus = GENERAL_STANDING_FOCUS
        res = self.smpl(body_pose=pose)  # last two hands joints are replaced by mono in SMPLH AND SMPLX

        all_joints = res.joints.squeeze(0).detach().cpu().numpy()
        human_pose = all_joints[SMPL_INDEX_FOR_OPENPOSE18]

        human_pose[:, 1] = -human_pose[:, 1]
        # human_pose[:, 2] = -human_pose[:, 2]
        # human_pose[:, 0] = -human_pose[:, 0]
        human_pose[2, 1] = 0.5*(human_pose[1, 1] + human_pose[2, 1])
        human_pose[5, 1] = 0.5*(human_pose[1, 1] + human_pose[5, 1])

        human_pose[3, :] = 0.5 * (human_pose[2, :] + human_pose[4, :])
        human_pose[6, :] = 0.5 * (human_pose[5, :] + human_pose[7, :])
        self.subset = np.arange(18).reshape([1, 18])
        self.crop_pose(crop_mode)
        self.pose_type = pose_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #  normalize
        pose_mask = self.subset[0] >= 0
        center = 0.5 * (np.max(human_pose[pose_mask, :], axis=0) + np.min(human_pose[pose_mask, :], axis=0))
        # max_size = np.max(np.max(human_pose[pose_mask, :], axis=0) - np.min(human_pose[pose_mask, :], axis=0))
        # human_pose = (human_pose - center) / max_size
        human_pose = human_pose - center

        # height_bias = np.array([0., 0.2, 0])
        # human_pose += height_bias

        # smpl render
        self.cvt_mask = torch.tensor([[[1., -1, 1, 1],
                                  [1, 1, -1, -1],
                                  [1, -1, 1, 1],
                                  [1, 1, 1, 1]]]).float().to(self.device)
        self.vertices = res.vertices.detach().to(self.device)
        normalize_center = (torch.amax(self.vertices, dim=-2, keepdim=True) + torch.amin(self.vertices, dim=-2, keepdim=True)) / 2.
        vertex_size = torch.max(torch.amax(self.vertices, dim=-2) - torch.amin(self.vertices, dim=-2))
        self.vertices = (self.vertices - normalize_center) / torch.tensor(vertex_size)
        # self.vertices[:, :, 2] = -self.vertices[:, :, 2]
        # self.vertices = self.vertices - normalize_center

        # 4x18
        human_pose = torch.tensor(human_pose).float() / torch.tensor(vertex_size)
        self.human_pose = torch.cat([human_pose, torch.ones(len(human_pose), 1)],
                                    dim=-1).T.float().to(self.device)

        self.faces = torch.tensor(self.smpl.faces.astype(np.int64), device=self.device)

        self.face_attributes = [
            torch.ones((1, len(self.faces), 3, 1), device=self.device)
        ]
        self.occulud_part_ind = [0, 14, 15, 16, 17]  # nose eyes and ears

    def get_mesh(self):
        # return self.vertices.cpu().numpy()[0], self.faces_org.cpu().numpy()
        return self.vertices.cpu().numpy()[0], self.faces.cpu().numpy()

    def render_occlusion(self, cam_pose, cam_k, HW, vis=False, **kwargs):
        fov_angles = -torch.arctan(cam_k[:, 1, 2] * 2 / cam_k[:, 1, 1] / 2).cpu().numpy() * 2.
        cam_projs = []
        for ang in fov_angles:
            cam_proj = kal.render.camera.generate_perspective_projection(ang, float(HW[1]) / HW[0])
            cam_projs.append(cam_proj)
            # print(ang, cam_proj)

        cam_projs = torch.stack(cam_projs, dim=0).to(self.device)
        # cam_proj = torch.tensor([[1.5, 1.5, 1.], [1., 1., 1.]], device='cuda').float()

        if self.pose_type == 'nerf':
            trans_cam_pos = cam_pose * self.cvt_mask

            trans_cam_pos[:, :, 0] = -trans_cam_pos[:, :, 0]
            trans_cam_pos[:, :3, :3] = trans_cam_pos[:, :3, :3].permute(0, 2, 1)
        else:
            trans_cam_pos = torch.eye(4).unsqueeze(0).repeat([len(cam_pose), 1, 1]).to(self.device)
            trans_cam_pos[:, :3, :3] = cam_pose[:, :3, :3]
            t = torch.linalg.inv(trans_cam_pos) @ cam_pose
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
        face_attributes = [torch.cat([i] * len(cam_pose)) for i in self.face_attributes]
        with_depth_feattures = face_attributes + [face_vertices_camera[:, :, :, -1:]] + [
            torch.cat([face_normals.unsqueeze(2)] * 3, dim=2)]
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            HW[0], HW[1], face_vertices_camera[:, :, :, -1],
            face_vertices_image, with_depth_feattures, face_normals[:, :, -1],
            rast_backend='cuda')

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        mask, depth_map_org, normal_map = image_features
        depth_map_org = torch.where(mask > 0.5, -depth_map_org, depth_map_org*0.+1000.)
        return depth_map_org

    def render(self, cam_pose, cam_k, HW, vis=False, **kwargs):
        depth_img = self.render_occlusion(cam_pose, cam_k, HW, vis=False, **kwargs)
        depth_img = depth_img.squeeze(-1)
        '''
        human_pose: 1x17x3,
        cam_pose: bx4x4,
        cam_k: bx3x3
        '''
        cur_camk = cam_k.clone()
        cur_camk[:, 0] *= (HW[1] / cur_camk[:, 0, 2, None] / 2.)
        cur_camk[:, 1] *= (HW[0] / cur_camk[:, 1, 2, None] / 2.)
        pose_in_cam = torch.matmul(torch.linalg.inv(cam_pose)[:, :3, :], self.human_pose)
        uvd = torch.matmul(cur_camk, pose_in_cam).permute([0, 2, 1])
        uv = uvd[:, :, :2] / (uvd[:, :, 2:] + 1e-8)

        unoccluded_subset = np.concatenate([self.subset] * len(cam_pose), axis=0)
        valid_pixel = 2
        depth_thresh = 0.1
        for batch in range(len(uvd)):
            for j in self.occulud_part_ind:  # 5 out of 18 joints
                depth_region = depth_img[batch,
                                  int(uv[batch][j][1]-valid_pixel):int(uv[batch][j][1]+valid_pixel),
                                  int(uv[batch][j][0] - valid_pixel):int(uv[batch][j][0] + valid_pixel)]
                if depth_region.numel() > 0:
                    depth = torch.min(depth_region)
                else:
                    depth = 1000
                joint_depth = -uvd[batch, j, -1]
                if j in [16, 17]:
                    if depth + depth_thresh+0.02 < joint_depth:
                        unoccluded_subset[batch, j] = -1
                else:
                    if depth + depth_thresh < joint_depth:
                        unoccluded_subset[batch, j] = -1

        bsz = uv.shape[0]
        pose_img = np.zeros([bsz, HW[0], HW[1], 3])
        for b in range(bsz):
            pose_img[b] = draw_bodypose(pose_img[b], uv[b].cpu().numpy(), unoccluded_subset) / 255.
            if vis:
                plt.imshow(pose_img[b])
                plt.show()
                plt.savefig(f'./pose_{b}.jpg')
        pose_img = torch.tensor(pose_img, device=self.device).float().permute([0, 3, 1, 2])
        # pose_img = torch.flip(pose_img, (3,))
        # depth_map_img = depth_img.unsqueeze(-1) - 0.2
        # depth_map_img /= 5.
        # depth_map_img = torch.clamp(depth_map_img, 0., 1.)
        # depth_map_img = depth_map_img.float().permute([0, 3, 1, 2])
        # depth_map = torch.cat([depth_map_img] * 3, dim=1)
        # # pose_img = depth_map*0.5+pose_img*0.5

        return pose_img

if __name__ == '__main__':
    from lib.load_data import angle_to_pose, angle_to_cam
    import imageio
    from human.smpl import DensePoseSMPL

    img_size = [512, 256]
    cam_k = torch.tensor([
        [500, 0, img_size[1]/2.],
        [0., 500., img_size[0]/2.],
        [0, 0, 1]
    ]).unsqueeze(0).to('cuda')
    # human = HumanPose('./human/3d_pose.npy', crop_mode='full', pose_type='nerf')
    # human = DensePoseSMPL(src_path='./human/smpl/', crop_mode='full', pose_type='nerf')
    human = HumanPoseSMPLX(src_path='./human/smpl/', crop_mode='full', pose_type='nerf')

    test_pose_num = 100
    pose_angles = np.deg2rad([(30., i * 360. / test_pose_num) for i in range(test_pose_num)])
    radius = 2.
    imgs = []
    for angle in pose_angles:
        theta, phi = angle
        # theta = np.pi - theta
        if human.pose_type == 'nerf':
            cam_pose = angle_to_pose(theta, phi, radius=radius, jitter=False, device='cuda', size=1)
        else:
            cam_pose = angle_to_cam([theta], [phi], radius=radius, jitter=False, device='cuda')

        img = human.render(cam_pose, cam_k, img_size, out_type='depth', vis=False)
        img = img.permute([0, 2, 3, 1])
        img = np.array(img.detach().cpu().numpy() * 255., np.uint8)
        # sav_img = Image.fromarray(img[0])
        # sav_img.save('./test.png')
        imgs.append(img[0])
        # imgs.append(img[1])
    imageio.mimwrite('video.rgb.mp4', imgs, fps=10, quality=8)

