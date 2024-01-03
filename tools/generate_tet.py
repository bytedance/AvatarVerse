import trimesh
import os
import numpy as np
from functools import reduce
tet_size_list = [
    [1, 7, 1],
    [1, 5, 1],
    [1, 3, 1],
    [1, 2, 1],
    [1, 1, 1],
]
out_path = 'data/tets/'
tet_num = 200**3

def convert_from_quartet_to_npz(quartetfile, npzfile):

    file1 = open(quartetfile, 'r')
    header = file1.readline()
    numvertices = int(header.split(" ")[1])
    numtets     = int(header.split(" ")[2])
    print(numvertices, numtets)

    # load vertices
    vertices = np.loadtxt(quartetfile, skiprows=1, max_rows=numvertices)
    # vertices = vertices - 0.5
    print(vertices.shape, vertices.min(), vertices.max())

    # load indices
    indices = np.loadtxt(quartetfile, dtype=int, skiprows=1+numvertices, max_rows=numtets)
    print(indices.shape)

    np.savez_compressed(npzfile, vertices=vertices, indices=indices)


for tet_size in tet_size_list:
    box_size = np.array(tet_size) / np.max(tet_size)
    box_size = box_size.tolist()
    m = trimesh.creation.box(extents=box_size)
    save_path = os.path.join(out_path, 'tet_%s.obj' % ('_'.join([str(i) for i in tet_size])))
    m.export(save_path)
    l_each_tet = round((reduce((lambda x, y: x * y), box_size) / float(tet_num)) ** (1./3), 5)

    out_put_tet = save_path.replace('.obj', '.tet')
    os.system('../quartet/quartet %s %f %s' % (save_path, l_each_tet, out_put_tet))
    convert_from_quartet_to_npz(out_put_tet, out_put_tet.replace('.tet', '.npz'))
    print('saved: ', out_put_tet.replace('.tet', '.npz'))

