import random

import numpy as np
import torch
from torch.utils.data import Dataset
from .load_llff import load_llff_data
# from .load_blender import load_blender_data
from .load_nsvf import load_nsvf_data
# from .load_blendedmvs import load_blendedmvs_data
# from .load_tankstemple import load_tankstemple_data
# from .load_deepvoxels import load_dv_data
# from .load_co3d import load_co3d_data
# from .load_nerfpp import load_nerfpp_data
from .diffusion_dataset import load_diffusion_data, load_diffusion_data_ff, load_diffusion_3d_data, \
    random_angles, angle_to_pose, safe_normalize, get_view_direction


def sample_camera_positions_with_angles(theta, phi, radius, device='cuda'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Phi is yaw in radians (-pi, pi)
    Theta is pitch in radians (0, pi)
    """
    phi = phi + np.pi
    n = len(phi)
    sample_r = radius.reshape([n, 1])
    theta = torch.clamp(theta, 1e-5, np.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    # sample_r = torch.rand((n, 1), device=device)
    # sample_r = sample_r * r[0] + (1 - sample_r) * r[1]

    compute_phi = -phi - 0.5 * np.pi

    output_points[:, 0:1] = sample_r * torch.sin(theta) * torch.cos(compute_phi)
    output_points[:, 2:3] = sample_r * torch.sin(theta) * torch.sin(compute_phi)
    output_points[:, 1:2] = sample_r * torch.cos(theta)
    rotation_angle = phi
    elevation_angle = theta
    return output_points, rotation_angle, elevation_angle


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_my_world2cam_matrix(forward_vector, origin, device='cuda'):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_noise = torch.randn_like(up_vector) * 0.02
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1) + up_noise)

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam


def angle_to_cam(theta, phi, radius, jitter, device):
    theta = torch.tensor(theta)
    phi = torch.tensor(phi)
    radius = torch.tensor([radius])
    camera_origin, rotation_angle, elevation_angle = sample_camera_positions_with_angles(theta, phi, radius, device)
    targets = 0.
    if jitter:
        camera_origin = camera_origin + (torch.rand_like(camera_origin) * 0.2 - 0.1)
        targets = targets + torch.randn_like(camera_origin) * 0.1
    forward_vector = normalize_vecs(camera_origin - targets)
    # Camera is always looking at the Origin point

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.001
    else:
        up_noise = 0.
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1) + up_noise)

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -camera_origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam_matrix = new_r @ new_t
    # world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device)
    return world2cam_matrix


class Diffusion3dDataset(Dataset):
    def __init__(self, training, device, args):
        super().__init__()
        self.training = training
        self.pose_type = args.pose_type
        self.dataset_type = args.dataset_type
        self.device = device
        # self.radius_scale = 2./3  # change the init bbox radius from 3 to 2, so the radius is scaled relatively
        self.radius_scale = 1/2.  # change the init bbox radius from 3 to 2, so the radius is scaled relatively
        self.radius_scale = 3/8.  # change the init bbox radius from 3 to 2, so the radius is scaled relatively
        # self.radius_scale = 1.
        # self.radius = 2.732 * self.radius_scale
        # self.radius_range = [0.8, 1.2]
        # self.radius_range = [1., 1.5]
        self.all_radius = np.array([
            [1.4, 2.1],
            [1., 1.5],
            [0.8, 1.2],
        ]) * 1.
        self.n_in_one = 1 if not hasattr(args, 'n_in_one') else args.n_in_one

        self.radius_range = self.all_radius[0] if training else self.all_radius[1]*1.1
        self.ff_angle_range = [-45., 45]
        self.pose_focus = [
            (torch.tensor([0., 0., 0], dtype=torch.float), 1.0)  # xyz bias, radius scale
        ]
        # if not training:
        #     self.pose_focus = [
        #         (torch.tensor([0., -0.05, 0], dtype=torch.float), 1.0)  # xyz bias, radius scale
        #     ]
        # self.fov_range = (50, 70)
        self.focal_range = args.focal_range if self.training else [0.7, 1.35]# 70-40 degree
        # self.focal_range = (0.7, 1.0)  # 70-50 degree
        if args.dataset_type == 'diffusion-ff':
            if training:
                # angles in radian
                self.pose_angles = random_angles(args.pose_size, theta_range=[80, 100], phi_range=[-75, 75], fix_view=False)
                # self.pose_angles = random_angles(args.pose_size, theta_range=[90, 90], phi_range=[0, 0], fix_view=False)
                # self.pose_angles = []
                # self.pose_angles += list(random_angles(args.pose_size, theta_range=[60, 120], phi_range=[-45, 45], fix_view=False))
                # self.pose_angles += list(random_angles(args.pose_size, theta_range=[60, 120], phi_range=[-60, 60], fix_view=False))
                # self.pose_angles += list(random_angles(args.pose_size, theta_range=[60, 120], phi_range=[-90, 90], fix_view=False))
                # np.random.shuffle(self.pose_angles)
                # self.pose_angles = self.pose_angles[:args.pose_size]
                # self.pose_angles = np.deg2rad([(90, i * 360. / args.pose_size-90) for i in range(args.pose_size)])
                # print(self.pose_angles)
                # self.radius_range = [1., 1]
            else:
                test_pose_num = 100
                max_theta = 10.
                max_phi = 60
                self.pose_angles = []
                for i in range(test_pose_num):
                    delta = np.pi * 2 * i / test_pose_num
                    theta = np.sin(delta) * max_theta + 90.
                    phi = np.cos(delta) * max_phi
                    self.pose_angles.append((theta, phi))
                self.pose_angles = np.deg2rad(self.pose_angles)
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-ff for %s' % ('training' if training else 'testing'), args.datadir)
        elif args.dataset_type == 'diffusion-round':
            if training:
                # angles in radian
                self.pose_angles = random_angles(args.pose_size, theta_range=[80, 100], phi_range=[-179, 179], fix_view=False)
            else:
                test_pose_num = min(100, args.pose_size)
                self.pose_angles = np.deg2rad([(80, i * 360. / test_pose_num) for i in range(test_pose_num)])
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-round for %s' % ('training' if training else 'testing'), args.datadir)
        elif args.dataset_type == 'diffusion-avatar':
            if training:
                # angles in radian
                self.pose_angles = random_angles(args.pose_size, theta_range=[60, 100], phi_range=[-179, 179], fix_view=False)
            else:
                test_pose_num = min(60, args.pose_size)
                self.pose_angles = np.deg2rad([(80, i * 360. / test_pose_num) for i in range(test_pose_num)])
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-avatar for %s' % ('training' if training else 'testing'), args.datadir)
        elif args.dataset_type == 'diffusion-avatar-full':
            if training:
                # angles in radian
                pose_angles_h = random_angles(args.pose_size*8//10, theta_range=[60, 100], phi_range=[-180, 180], fix_view=False)
                pose_angles_top = random_angles(args.pose_size * 2 // 10, theta_range=[20, 60], phi_range=[-180, 180], fix_view=False)
                self.pose_angles = np.concatenate([pose_angles_h, pose_angles_top], axis=0)
            else:
                test_pose_num = min(100, args.pose_size)
                self.pose_angles = np.deg2rad([(80, i * 360. / test_pose_num) for i in range(test_pose_num)])
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-avatar for %s' % ('training' if training else 'testing'), args.datadir)
        elif args.dataset_type == 'diffusion-3d':
            if self.training:
                # angles in radian
                self.pose_angles = random_angles(args.pose_size, theta_range=[10., 100], phi_range=[0, 360], fix_view=False)
            else:
                test_pose_num = min(100, args.pose_size)
                self.pose_angles = np.deg2rad([(80, i * 360. / test_pose_num) for i in range(test_pose_num)])
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-3d for %s' % ('training' if training else 'testing'), args.datadir)

        elif args.dataset_type == 'diffusion-multi-diffusion':
            # if self.training:
            #     # angles in radian
            #     self.pose_angles = random_angles(args.pose_size, theta_range=[10., 100], phi_range=[0, 360], fix_view=False)
            # else:
            #     test_pose_num = min(100, args.pose_size)
            #     self.pose_angles = np.deg2rad([(60, i * 360. / test_pose_num) for i in range(test_pose_num)])
            self.radius_range = self.all_radius[1] * 1.1
            if self.training:
                phi_range = [0, 360 // self.n_in_one]
                self.pose_angles = random_angles(args.pose_size, theta_range=[80, 100], phi_range=phi_range, fix_view=False)
                self.pose_angles = [[np.pi/2, 0]]
                self.radius_range[0] = self.radius_range[1] = 0.5*(self.radius_range[0]+self.radius_range[1])
                self.focal_range[0] = self.focal_range[1] = 0.5*(self.focal_range[0]+self.focal_range[1])
            else:
                test_pose_num = min(100, args.pose_size)
                self.pose_angles = np.deg2rad([(80, i * 360. / test_pose_num) for i in range(test_pose_num)])
                self.n_in_one = 1
            self.near, self.far = 1. * self.radius_scale, 2.732 * self.radius_scale  # designed for cube radius 1
            print('Loaded diffusion-3d for %s' % ('training' if training else 'testing'))

        else:
            raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

        # self.radius = 2. * self.radius_scale
        self.dir_text = args.dir_text if self.n_in_one <= 1 else False
        self.H = args.height
        self.W = args.width

        self.max_batch_size = args.max_batch_size

    def __len__(self):
        return max(self.max_batch_size, len(self.pose_angles))

    def set_radius(self, radius_stage):
        if radius_stage >= len(self.all_radius):
            radius_stage = len(self.all_radius) - 1

        print('\nradius range set: ', self.radius_range, '->', self.all_radius[radius_stage])
        self.radius_range = self.all_radius[radius_stage]

    def set_radius_range(self, radius_range):
        self.radius_range = radius_range

    def set_focus(self, new_focus):
        if len(self.pose_focus) == 1:
            # self.pose_focus = [
            #     (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
            #     (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
            #     (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
            #     (torch.tensor([0., -0.4, 0], dtype=torch.float), 0.3),  # head
            #     (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot
            #     # big pose, hand focus
            #     (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
            #     (torch.tensor([-0.4, -0.3, 0], dtype=torch.float), 0.3),  # close right hand
            # ]
            self.pose_focus = new_focus
            print('\n start focus pose')

    def set_angle(self, amgle_stage):
        return
        if amgle_stage == 1:
            ff_angle_range = [-60, 60]
        elif amgle_stage == 2:
            ff_angle_range = [-90, 90]
        else:
            ff_angle_range = self.ff_angle_range
        print('\nff angle range set: ', self.ff_angle_range, '->', ff_angle_range)
        self.ff_angle_range = ff_angle_range
        self.pose_angles = random_angles(len(self.pose_angles), theta_range=[60, 120], phi_range=self.ff_angle_range,
                                         fix_view=False)

    def __getitem__(self, idx):
        idx = idx % len(self.pose_angles)
        theta, phi = self.pose_angles[idx]
        # theta = torch.tensor([theta], dtype=torch.float, device=self.device)
        # phi = torch.tensor([phi], dtype=torch.float, device=self.device)
        theta = np.array([theta])
        phi = np.array([phi])
        if self.training:
            bias, radius_scale = random.choice(self.pose_focus)
            bias = bias.to(self.device)
            # bias = torch.tensor([0., 1., 0], dtype=torch.float) * np.random.randn() * 0.2
            # fov = np.random.random() * (self.fov_range[1] - self.fov_range[0]) + self.fov_range[0]
            focal_mul = np.random.random() * (self.focal_range[1] - self.focal_range[0]) + self.focal_range[0]
            random_radius = np.random.random() * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
            focal_mul /= radius_scale
            # fov = (self.fov_range[1] + self.fov_range[0]) / 2
            # focal = W / (2 * np.tan(np.deg2rad(train_focal) / 2))
            # random_radius = (np.random.random() * 0.4 + 0.8) * self.radius
            # random_radius = self.radius
        else:
            bias, radius_scale = self.pose_focus[0]
            bias = bias.to(self.device)
            # fov = (self.fov_range[1] + self.fov_range[0]) / 2
            focal_mul = (self.focal_range[1] + self.focal_range[0]) / 2 / radius_scale
            random_radius = (self.radius_range[1] + self.radius_range[0]) / 2
            # random_radius = self.radius

        n_in_one_pose = []
        n_in_one_dir_text = []
        n_in_one_light_pos = []
        n_in_one_k = []
        for i in range(self.n_in_one):
            focal = self.H * focal_mul
            phi_view = phi + np.pi*2/self.n_in_one*i
            k = torch.tensor([[
                [focal, 0, 0.5 * self.W],
                [0, focal, 0.5 * self.H],
                [0, 0, 1]
            ]])
            if self.pose_type == 'nerf':
                pose = angle_to_pose(theta, phi_view, radius=random_radius,
                                     jitter=False, # self.training and len(self.pose_focus) == 1,
                                     device=self.device, size=1)
            else:
                pose = angle_to_cam(theta, phi_view, radius=random_radius,
                                    jitter=False, # self.training and len(self.pose_focus) == 1,
                                    device=self.device)
            pose[:, :3, 3] += bias
            if self.dir_text:
                dir_text, dir_names = get_view_direction(theta, phi_view, overhead=np.deg2rad(30), front=np.deg2rad(60))
                dir_text = dir_text.to(self.device)
            else:
                dir_text = torch.zeros(1, dtype=torch.long).to(self.device)
                dir_names = ''

            # print('-----\n pose and dir text ', idx, theta/np.pi*180, phi/np.pi*180, dir_text, dir_names, '\n-------')
            # train light position
            pos = pose[:, :3, 3] if self.pose_type == 'nerf' else -pose[:, :3, 3]
            if self.training:
                # light_dir = torch.randn([len(pos), 3], device=self.device) / 6. + pos
                light_dir = torch.rand([3], device=self.device)*0.4-0.2 + torch.tensor([0., 0.5, 0.], device=self.device).reshape([1, 3]) + pos

                light_radius = (torch.rand([len(pos), 1], device=self.device) * 0.6 + 1.0) * self.radius_range[0]
                light_pos = safe_normalize(light_dir) * light_radius
            else:
                light_dir = torch.tensor([0., 0.5, 0.], device=self.device).reshape([1, 3]) + pos
                light_radius = self.radius_range[0]
                light_pos_test = safe_normalize(light_dir) * light_radius
                light_pos = light_pos_test.repeat([len(pos), 1])
            n_in_one_pose.append(pose)
            n_in_one_dir_text.append(dir_text)
            n_in_one_light_pos.append(light_pos)
            n_in_one_k.append(k)
        n_in_one_pose = torch.cat(n_in_one_pose, dim=0)
        n_in_one_dir_text = torch.cat(n_in_one_dir_text, dim=0)
        n_in_one_light_pos = torch.cat(n_in_one_light_pos, dim=0)
        n_in_one_k = torch.cat(n_in_one_k, dim=0)
        n_in_one_cam = torch.tensor([theta, phi]).to(k)
        # return pose.squeeze(0), dir_text.squeeze(0), light_pos.squeeze(0), k.squeeze(0)
        if self.dataset_type == 'diffusion-multi-diffusion':
            return n_in_one_pose.squeeze(0), n_in_one_dir_text.squeeze(0), n_in_one_light_pos.squeeze(0), n_in_one_k.squeeze(0), n_in_one_cam.squeeze(0)
        else:
            return n_in_one_pose.squeeze(0), n_in_one_dir_text.squeeze(0), n_in_one_light_pos.squeeze(0), n_in_one_k.squeeze(0)

