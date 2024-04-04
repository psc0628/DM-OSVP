import os
import sys
import trimesh
# import mesh2sdf
from mesh_to_sdf import mesh_to_voxels
import trimesh
import skimage
import numpy as np
import time
# import aspose.threed as a3d
import open3d as o3d
# import pymeshlab as ml

# make sure input with obj file
filename_obj = sys.argv[1]
if not filename_obj.endswith('.obj'):
    print('Please input a file with the .obj extension')
    exit(-1)

resolution = sys.argv[2]
if int(resolution) <= 0:
    print('Please input resolution with an postive interger')
    exit(-1)

filp_y = sys.argv[3]
if filp_y != '1' and filp_y != '0' and filp_y != '2' and filp_y != '3':
    print('Please input flip y or not with "1" or "0" or "2" or "3"')
    exit(-1)

# assume we get obj
current_folder = os.path.dirname(os.path.abspath(__file__))

cmd = 'python mesh_sampling_geo_color_shapenet.py '
cmd += current_folder + '/' + filename_obj + ' '
cmd += current_folder + '/' + filename_obj[:filename_obj.find('/')] + '/ '
cmd += '--cloudcompare_bin_path D:\Software\CloudCompare\CloudCompare.exe '
cmd += '--target_points 500000 '
cmd += '--resolution 1024 '
os.system(cmd)
if os.path.exists(current_folder + '/' + filename_obj.replace('.obj','.ply')):
    os.remove(current_folder + '/' + filename_obj.replace('.obj','.ply'))
os.rename(current_folder + '/' + filename_obj[:filename_obj.find('/')] +'/vox_sampled_mesh_0.ply', current_folder + '/' + filename_obj.replace('.obj','.ply'))

# 加载点云
filename = filename_obj
size = int(resolution)
print('voxel resolution is', size)

t0 = time.time()
pcd = o3d.io.read_point_cloud(filename.replace('.obj', '.ply'))
# 将 Vector3dVector 转换为 NumPy 数组
points_array = np.asarray(pcd.points)
# 交换 Z 和 Y 轴
points_array[:, [1, 2]] = points_array[:, [2, 1]]
# 根据需求翻转 X 和 Y 轴
if filp_y == '0':
    points_array[:, 0] = -points_array[:, 0]
elif filp_y == '1':
    points_array[:, 1] = -points_array[:, 1]
elif filp_y == '2':
    points_array[:, [0, 1]] = points_array[:, [1, 0]]
elif filp_y == '3':
    points_array[:, [0, 1]] = points_array[:, [1, 0]]
    points_array[:, 0] = -points_array[:, 0]
    points_array[:, 1] = -points_array[:, 1]

# 标准化点云
center = np.mean(points_array, axis=0)
radius = np.max(np.linalg.norm(points_array - center, axis=1))
scale = 1.0 / radius
points_array = (points_array - center) * scale

# 将 NumPy 数组转换回 Vector3dVector
pcd.points = o3d.utility.Vector3dVector(points_array)
# Save the sampled points to a PCD file
o3d.io.write_point_cloud(filename[:-4] + '.pcd', pcd)
t1 = time.time()
print('It takes %.4f seconds to obtain %s' % (t1-t0, filename[:-4] + ' pcd'))

