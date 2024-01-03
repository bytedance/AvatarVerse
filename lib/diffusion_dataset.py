import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


from packaging import version as pver
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

DIR_COLORS = np.array([
    [255, 0, 0, 255],  # front  red
    [0, 255, 0, 255],  # side  green
    [0, 0, 255, 255],  # back   blue
    [255, 255, 0, 255],  # side yellow
    [255, 0, 255, 255],  # overhead  purple
    [0, 255, 255, 255],  # bottom   cyan
], dtype=np.uint8)


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=0.01)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        center = (a +b+c+d)/4. - pos
        center = center / np.linalg.norm(center) * np.linalg.norm(pos) + pos

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, center]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    direct_names = np.array(['front', 'side', 'back', 'side', 'top', 'botoom'])
    res = np.zeros(thetas.shape[0], dtype=np.long)
    phis = phis % (np.pi * 2)
    # first determine by phis
    front = front * 0.5
    res[(phis < front) & (phis >= 2*np.pi - front)] = 0
    res[(phis >= front) & (phis < np.pi - front)] = 1
    res[(phis >= np.pi - front) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front)) & (phis < 2*np.pi - front)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5

    res_tensor = torch.tensor(res, dtype=torch.long)
    return res_tensor, direct_names[res]


def angle_to_pose(theta, phi, radius, jitter, device, size=1):
    theta_center = torch.tensor(np.pi - theta, dtype=torch.float, device=device)
    phi_center = torch.tensor(phi, dtype=torch.float, device=device)
    if not isinstance(radius, list):
        radius = [radius] * size
    else:
        assert len(radius) == size
    radius = torch.FloatTensor(radius).to(device)
    centers = torch.stack([
        radius * torch.sin(theta_center) * torch.sin(phi_center),
        radius * torch.cos(theta_center),
        radius * torch.sin(theta_center) * torch.cos(phi_center),
    ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.1

    # lookat
    forward_vector = -safe_normalize(targets - centers)
    tmp_up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, tmp_up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(tmp_up_vector) * 0.001
    else:
        up_noise = 0
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def random_angles(size, theta_range=[60, 100], phi_range=[-180, 180], fix_view=False):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    if fix_view:
        thetas = np.deg2rad([90])
        phis_can = -np.deg2rad([0, 10, -10, 30, -30, 80, 100, 180, 170, 190, 150, 210, -80, -100])
        phis = np.random.choice(phis_can, size)
    else:
        half = size // 2

        mind = np.cos(theta_range[1])
        maxd = np.cos(theta_range[0])
        thetas = np.arccos(np.random.random(half) * (maxd - mind) + mind)

        thetas = np.concatenate(
            [thetas,
             np.random.random(size - half) * (theta_range[1] - theta_range[0]) + theta_range[0]])

        # # Increase proportion of angles near 0 and 180 degrees for theta with some randomness
        extra_angles_range = 5  # +/- 5 degrees around 0 and 180
        extra_angles_size = int(size * 0.1)  # 10% of the total size
        extra_thetas_0 = np.random.uniform(360 - extra_angles_range, 360 + extra_angles_range, extra_angles_size)
        extra_thetas_0[extra_thetas_0 > 360] -= 360
        # extra_thetas_180 = np.random.uniform(180 - extra_angles_range, 180 + extra_angles_range, extra_angles_size // 2)
        # extra_thetas = np.deg2rad(np.concatenate([extra_thetas_0, extra_thetas_180]))
        extra_thetas = np.deg2rad(extra_thetas_0)
        phis = np.concatenate([extra_thetas,
            np.random.random(size - extra_angles_size) * (phi_range[1] - phi_range[0]) + phi_range[0]])
        # phis =  np.random.random(size) * (phi_range[1] - phi_range[0]) + phi_range[0]

    # print(thetas)
    # print(phis)

    return np.stack([thetas, phis], axis=-1)


def rand_poses(size, device, radius=1., theta_range=[30, 100], phi_range=[0, 360], return_dirs=False,
               angle_overhead=30, angle_front=60, jitter=False, fix_view=False):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    if not isinstance(radius, list):
        radius = [radius] * size
    else:
        assert len(radius) == size
    radius = torch.tensor(radius, device=device)
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    # radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    # radius = 1.
    if fix_view:
        # thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        thetas = torch.tensor(np.deg2rad([60]), dtype=torch.float, device=device)
        # phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis_can = -np.deg2rad([0, 10, -10, 30, -30, 80, 100, 180, 170, 190, 150, 210, -80, -100])
        # phis_can = np.deg2rad([0])
        phis = torch.tensor(np.random.choice(phis_can, size), dtype=torch.float, device=device)
        # phis += torch.rand(size, device=device) * np.deg2rad(5.)
    else:
        # mind=np.cos(theta_range[1])
        # maxd=np.cos(theta_range[0])
        # thetas = np.arccos(torch.rand(size, device=device) * (maxd - mind) + mind)
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # thetas = torch.tensor(np.deg2rad([60]), dtype=torch.float, device=device)
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        # phis_can = np.deg2rad([0, 10, -10, 30, -30, -45, -60, -70, -80, -90, -100, 80, 100, 170, 150, 210])
        # phis_can[phis_can<0] += np.pi * 2
        # # phis_can = np.deg2rad([0, 180])
        # phis = torch.tensor(np.random.choice(phis_can, size), dtype=torch.float, device=device)
        # phis += torch.rand(size, device=device) * np.deg2rad(5.)
    theta_cvt = np.pi - thetas
    centers = torch.stack([
        radius * torch.sin(theta_cvt) * torch.sin(phis),
        radius * torch.cos(theta_cvt),
        radius * torch.sin(theta_cvt) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = -safe_normalize(targets - centers)
    tmp_up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, tmp_up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(tmp_up_vector) * 0.02
    else:
        up_noise = 0
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = torch.zeros(thetas.shape[0], dtype=torch.long)

    return poses, dirs.to(device)


def circle_poses(device, radius=1., theta=60., phi=0., return_dirs=False, angle_overhead=30., angle_front=60.):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    theta_cvt = np.pi - thetas

    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(theta_cvt) * torch.sin(phis),
        radius * torch.cos(theta_cvt),
        radius * torch.sin(theta_cvt) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = torch.zeros(thetas.shape[0], dtype=torch.long)

    return poses, dirs.to(device)


def face_forward_poses(device, size, radius=1., circle_num=3, front_rate=0.5, max_theta=10., max_phi=30, return_dirs=False, angle_overhead=30., angle_front=60.):
    front_circle_num = int((1-front_rate) * size) // circle_num * circle_num

    front_num = size - front_circle_num
    if not isinstance(radius, list):
        radius = [radius] * size
    else:
        assert len(radius) == size

    def get_pose(theta, phi, radius, angle_overhead=30., angle_front=60.):

        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)

        thetas = torch.FloatTensor([theta]).to(device)
        phis = torch.FloatTensor([phi]).to(device)

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)  # [B, 3]

        # lookat
        forward_vector = safe_normalize(centers)
        up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

        poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
        poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
        poses[:, :3, 3] = centers

        if return_dirs:
            dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
        else:
            dirs = torch.zeros(thetas.shape[0], dtype=torch.long)

        return poses, dirs.to(device)

    all_poses = []
    all_dirs = []
    ind = 0
    for _ in range(front_num):
        front_pose, dirs = get_pose(90., 0, radius[ind], angle_overhead=angle_overhead, angle_front=angle_front)
        all_poses.append(front_pose)
        all_dirs.append(dirs)
        ind += 1

    front_circle_min_degree = 10,
    degrees_theta = np.linspace(front_circle_min_degree, max_theta, circle_num).flatten()
    degrees_phi = np.linspace(front_circle_min_degree, max_phi, circle_num).flatten()
    if circle_num < 2:
        degrees_theta = [max_theta]
        degrees_phi = [max_phi]
    num_each_circle = front_circle_num // circle_num
    for t, p in zip(degrees_theta, degrees_phi):
        for i in range(num_each_circle):
            delta = np.pi * 2 * i / num_each_circle
            theta = np.sin(delta) * t + 90.
            phi = np.cos(delta) * p
            pose, dirs = get_pose(theta, phi, radius[ind], angle_overhead=angle_overhead, angle_front=angle_front)
            ind += 1
            all_poses.append(pose)
            all_dirs.append(dirs)
    all_poses = torch.cat(all_poses, dim=0)
    all_dirs = torch.cat(all_dirs, dim=0)
    return all_poses, all_dirs


def lego_data():
    scene_scale = 2 / 3
    all_c2w = []

    split_name = "train"
    root = '../data/lego/'
    data_path = os.path.join(root, split_name)
    data_json = os.path.join(root, "transforms_" + split_name + ".json")

    print("LOAD DATA", data_path)

    j = json.load(open(data_json, "r"))

    # OpenGL -> OpenCV
    cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
    cam_rot = torch.tensor([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for frame in tqdm.tqdm(j["frames"]):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w = c2w @ cam_trans  # To OpenCV
        c2w = cam_rot @ c2w
        all_c2w.append(c2w)
        # break
    # focal = float(
    #     0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
    # )
    c2w = torch.stack(all_c2w)
    c2w[:, :3, 3] *= scene_scale
    return c2w


def load_diffusion_data(num, radius, dir_text=True, H=512, W=512, fov_range=(50, 70)):
    fov = (fov_range[1] + fov_range[0]) / 2
    focal = W / (2 * np.tan(np.deg2rad(fov) / 2))
    # train pose
    poses_train, text_with_dirs_train = rand_poses(num, 'cpu', return_dirs=dir_text, radius=radius)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    # test pose
    poses_test = []
    text_with_dirs_test = []
    test_pose_num = 100
    for i in range(test_pose_num):
        phi = i * 360. / test_pose_num
        poses, text_with_dirs = circle_poses('cpu', theta=60, phi=phi, return_dirs=dir_text, radius=radius)
        poses_test.append(poses)
        text_with_dirs_test.append(text_with_dirs)

    poses_test = torch.cat(poses_test, dim=0)
    text_with_dirs_test = torch.cat(text_with_dirs_test, dim=0)
    all_texts = torch.cat([text_with_dirs_train, text_with_dirs_test], dim=0)
    all_poses = torch.cat([poses_train, poses_test], dim=0)
    splits = (np.arange(num), np.arange(test_pose_num) + num, np.arange(test_pose_num) + num)
    return all_texts.numpy(), all_poses.numpy(), poses_test.numpy(), text_with_dirs_test.numpy(), [H, W, focal], splits


def load_diffusion_data_ff(num, radius, dir_text=True, H=512, W=512, fov_range=(50, 70)):
    fov = (fov_range[1] + fov_range[0]) / 2
    focal = W / (2 * np.tan(np.deg2rad(fov) / 2))
    test_num = min(100, num)

    train_focal = np.random.random(num) * (fov_range[1] - fov_range[0]) + fov_range[0]
    train_focal = W / (2 * np.tan(np.deg2rad(train_focal) / 2))
    Ks = []
    for f in train_focal:
        k = np.array([
            [f, 0, 0.5 * W],
            [0, f, 0.5 * H],
            [0, 0, 1]
        ])
        Ks.append(k)
    Ks += [np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])] * test_num
    Ks = np.stack(Ks, axis=0)
    # train pose
    radius_range = [0.8, 1.2]
    random_radius = ((np.random.random(num) * (radius_range[1] - radius_range[0]) + radius_range[0]) * radius).tolist()
    poses, pose_dirs = rand_poses(num, 'cpu', radius=random_radius, theta_range=[80, 100], phi_range=[-89, 89], return_dirs=dir_text)
    # poses, pose_dirs = face_forward_poses('cpu', num, radius=random_radius, circle_num=5, front_rate=0.4, max_circle_degree=60., return_dirs=dir_text)
    # test pose
    poses_test, pose_dirs_test = face_forward_poses('cpu', test_num, radius=radius*1., circle_num=1, front_rate=0., max_theta=10., max_phi=60, return_dirs=dir_text)
    poses = torch.cat([poses, poses_test], dim=0)
    pose_dirs = torch.cat([pose_dirs, pose_dirs_test], dim=0)
    splits = (np.arange(num), np.arange(test_num) + num, np.arange(test_num) + num)
    return pose_dirs.numpy(), poses.numpy(), poses.numpy(), pose_dirs.numpy(), [H, W, focal], splits, Ks


def load_diffusion_3d_data(num, radius, dir_text=True, H=512, W=512, fov_range=(50, 70)):
    fov = (fov_range[1] + fov_range[0]) / 2
    focal = W / (2 * np.tan(np.deg2rad(fov) / 2))

    train_focal = np.random.random(num) * (fov_range[1] - fov_range[0]) + fov_range[0]
    train_focal = W / (2 * np.tan(np.deg2rad(train_focal) / 2))
    Ks = []
    for f in train_focal:
        k = np.array([
            [f, 0, 0.5 * W],
            [0, f, 0.5 * H],
            [0, 0, 1]
        ])
        Ks.append(k)

    # train pose
    random_radius = ((np.random.random(num) * 0.4 + 0.8) * radius).tolist()
    poses_train, text_with_dirs_train = rand_poses(num, 'cpu', radius=random_radius, theta_range=[5., 100], phi_range=[0, 360], return_dirs=dir_text)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    # train light position
    pos = poses_train[:, :3, 3]
    light_dir = torch.randn([len(pos), 3], device='cpu')/2. + pos
    light_radius = (torch.rand([len(pos), 1], device='cpu') * 0.6 + 0.6) * radius
    light_pos_train = safe_normalize(light_dir) * light_radius
    # test pose
    poses_test = []
    text_with_dirs_test = []
    test_pose_num = 100
    for i in range(test_pose_num):
        phi = i * 360. / test_pose_num
        poses, text_with_dirs = circle_poses('cpu', theta=60, phi=phi, return_dirs=dir_text, radius=radius)
        poses_test.append(poses)
        text_with_dirs_test.append(text_with_dirs)
    Ks += [np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])] * test_pose_num
    Ks = np.stack(Ks, axis=0)
    poses_test = torch.cat(poses_test, dim=0)

    # test light position
    pos = poses_test[:, :3, 3]
    light_dir = torch.tensor([0.5, 0.5, 0.5], device='cpu').reshape([1, 3]) + pos * 0.8
    light_radius = 0.8 * radius
    light_pos_test = safe_normalize(light_dir) * light_radius
    light_pos_test = light_pos_test.repeat([len(pos), 1])

    text_with_dirs_test = torch.cat(text_with_dirs_test, dim=0)
    all_texts = torch.cat([text_with_dirs_train, text_with_dirs_test], dim=0)
    all_poses = torch.cat([poses_train, poses_test], dim=0)
    all_light = torch.cat([light_pos_train, light_pos_test], dim=0)
    splits = (np.arange(num), np.arange(test_pose_num) + num, np.arange(test_pose_num) + num)
    return all_texts.numpy(), all_poses.numpy(), poses_test.numpy(), all_light.numpy(), text_with_dirs_test.numpy(), [H, W, focal], splits, Ks


class NeRFDataset:
    def __init__(self, device, dir_text=True, type='train', H=512, W=512, size=100, fov_range=(50, 70), fixed_view_seq=100):
        super().__init__()

        self.dir_text = dir_text
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.size = size
        self.fov_range = fov_range
        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2
        self.radius = 2.1
        fov = (self.fov_range[1] + self.fov_range[0]) / 2
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        fix_intrinsics = np.array([focal, focal, self.cx, self.cy])

        self.view_seq = []

        if not self.training and fixed_view_seq > 0:
            for i in range(fixed_view_seq):
                phi = 360. * float(i) / float(fixed_view_seq)
                poses, dirs = circle_poses(self.device, theta=60, phi=phi, return_dirs=self.dir_text, radius=self.radius)
                # sample a low-resolution but full image for CLIP
                rays = get_rays(poses, fix_intrinsics, self.H, self.W, -1)

                data = {
                    'H': self.H,
                    'W': self.W,
                    'rays_o': rays['rays_o'],
                    'rays_d': rays['rays_d'],
                    'dir': dirs,
                    'pose': poses,
                    'intrinsics': fix_intrinsics
                }
                self.view_seq.append(data)
                # all_poses.append(poses)
                # all_dirs.append(dirs * 0 + 0)
        # [debug] visualize poses
        # all_poses = []
        # all_dirs = []
        # poses, dirs = rand_poses(100, self.device, return_dirs=self.dir_text, radius=3.)
        # all_poses.append(poses)
        # all_dirs.append(dirs*0+1)
        # # #
        # poses = lego_data()
        # dirs = torch.zeros([len(poses)], dtype=torch.long) + 2
        # all_poses.append(poses)
        # all_dirs.append(dirs)
        # all_poses = torch.cat(all_poses, dim=0)
        # all_dirs = torch.cat(all_dirs, dim=0)
        # visualize_poses(all_poses.detach().cpu().numpy(), all_dirs.detach().cpu().numpy())

    def collate(self, index):

        B = len(index)  # always 1

        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, return_dirs=self.dir_text, radius=self.radius)

            # random focal
            # fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            fov = (self.fov_range[1] + self.fov_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])
            # sample a low-resolution but full image for CLIP
            rays = get_rays(poses, intrinsics, self.H, self.W, -1)

            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'dir': dirs,
                'pose': poses,
                'intrinsics': intrinsics
            }
        else:
            if len(self.view_seq) == 0:
                # circle pose
                phi = index[0] * 360 / self.size
                poses, dirs = circle_poses(self.device, theta=60, phi=phi, return_dirs=self.dir_text, radius=self.radius)

                # fixed focal
                fov = (self.fov_range[1] + self.fov_range[0]) / 2
                focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
                intrinsics = np.array([focal, focal, self.cx, self.cy])
                # sample a low-resolution but full image for CLIP
                rays = get_rays(poses, intrinsics, self.H, self.W, -1)

                data = {
                    'H': self.H,
                    'W': self.W,
                    'rays_o': rays['rays_o'],
                    'rays_d': rays['rays_d'],
                    'dir': dirs,
                    'pose': poses,
                    'intrinsics': intrinsics
                }
            else:
                ind = int(float(index[0]) * len(self.view_seq) / self.size)
                data = self.view_seq[ind]

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        return loader


if __name__ == '__main__':
    d = NeRFDataset('cpu', True, 'test', size=6, fixed_view_seq=6)