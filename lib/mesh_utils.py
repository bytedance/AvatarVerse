# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import xatlas
import cv2
import PIL
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import time
import os
import trimesh
from PIL import Image
import kaolin as kal


# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def xatlas_uvmap(ctx, mesh_v, mesh_pos_idx, resolution):
    vmapping, indices, uvs = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_pos_idx.detach().cpu().numpy())

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
    mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), (resolution, resolution))

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
    mask = rast[..., 3:4] > 0
    return uvs, mesh_tex_idx, gb_pos, mask


def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, tex_map, save_path):
    import os

    matname = os.path.join(save_path, 'model.mtl')
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd texture.png\n')
    fid.close()
    ####

    modelname = os.path.join(save_path, 'model.obj')
    fid = open(modelname, 'w')
    fid.write('mtllib model.mtl\n')

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    # texture map
    lo, hi = (0, 1)
    img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32).squeeze(0)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(os.path.join(save_path, 'texture.png'))

    return



def init_voxel(mesh_path, reso, grid, value=1., ratio=0.8):
    xx, yy, zz = voxelize_mesh_2(mesh_path, reso, ratio=ratio)
    print(xx.shape)
    grid.grid.data[:, :, xx, yy, zz] = value
    # grid.density_data.data[target_ind.long()] = value


def voxelize_mesh_2(mesh_path, reso, ratio):
    round_ratio = round(ratio, 1)
    voxel_grid_path = mesh_path.replace('.obj', '_voxel_grid_%s_%.1f.npy' % ('-'.join([str(i) for i in reso]), round_ratio))
    if not os.path.exists(voxel_grid_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

        max_size = np.max(max_bound - min_bound) / ratio
        center = (min_bound + max_bound) * 0.5
        min_bound = center - max_size / 2.
        max_bound = center + max_size/ 2.
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
        np.save(voxel_grid_path, occupancy)
    else:
        occupancy = np.load(voxel_grid_path)
    # print(occupancy.shape)
    xx, yy, zz = np.nonzero(occupancy)
    # xx, yy, zz = zz, xx, yy
    # xx, zz = reso[0] - zz, xx
    xx, yy, zz = yy, reso[0] - xx, zz
    print(np.max(xx), np.max(yy), np.max(zz))
    return xx, yy, zz

def fix_obj_holes(obj_path, out_path):

    mesh = trimesh.load(obj_path)

    if not mesh.is_watertight:
        # Get the list of holes
        holes = mesh.holes
        # Get the vertices and faces of the mesh
        vertices = mesh.vertices
        faces = mesh.faces
        # Iterate over the holes
        for hole in holes:
            # Create a new triangle fan to fill the hole
            fan = []
            for i, index in enumerate(hole):
                if i < len(hole) - 1:
                    # Add a triangle for each pair of adjacent vertices in the hole
                    fan.append([index, hole[i + 1], len(vertices)])
                else:
                    # Add a triangle for the last vertex and the first vertex in the hole
                    fan.append([index, hole[0], len(vertices)])
                # Add the new vertex to the list of vertices
                vertices = np.vstack([vertices, np.mean(vertices[hole], axis=0)])
            # Add the new faces to the list of faces
            faces = np.vstack([faces, fan])
        # Create a new mesh with the filled holes
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(out_path)


def remap_uv_map(src_obj, dst_obj, src_uvmap):
    mesh_src = trimesh.load(src_obj, process=False, include_tex=True)

    texture_image = Image.open(src_uvmap)
    texture_image_data = np.array(texture_image)

    mesh_src.visual.material.image = texture_image_data
    mesh_src.visual.material.uv = mesh_src.visual.uv
    # tex = trimesh.visual.TextureVisuals(image=im)
    # mesh_src.visual.texture = tex
    # mesh_src.show()

    scene = mesh_src.scene()
    scene.show()
    mesh_dst = trimesh.load(dst_obj)
    print()


def load_obj_kaolin(obj_path, src_uvmap):
    mesh = kal.io.obj.import_mesh(obj_path, with_materials=True)
    vertices = mesh.vertices
    faces = mesh.faces
    uvs = mesh.uvs
    face_uvs_idx = mesh.face_uvs_idx
    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx)

    texture_image = Image.open(src_uvmap)
    texture_image_data = np.array(texture_image)

    # render
    cam_pos = torch.eye(4).unsqueeze(0)
    cam_pos[:, 2, 3] = -1.5
    rot = cam_pos[..., :3, :3].cuda().float()
    trans = cam_pos[..., :3, 3].cuda().float()
    cam_proj = torch.tensor([1.5, 1.5, 1.], device='cuda').float()
    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            vertices,
            faces, cam_proj,
            camera_rot=rot, camera_trans=trans
        )

    face_attributes = [
        face_uvs,
        torch.ones((1, faces.shape[0], 3, 1), device='cuda')
    ]
    with_depth_feattures = self.face_attributes + [face_vertices_camera[:, :, :, -1:]]
    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        self.render_size, self.render_size, face_vertices_camera[:, :, :, -1],
        face_vertices_image, with_depth_feattures, face_normals[:, :, -1],
        rast_backend='cuda')




if __name__  == '__main__':
    remap_uv_map('./objs/head/mean_head3d_uv.obj',
                 './objs/head/mean_head3d_uv_watertight.obj',
                 './objs/head/uvmap.png')