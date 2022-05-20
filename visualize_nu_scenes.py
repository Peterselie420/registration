import copy
import itertools
import math

import open3d as o3d

import data_nu_scenes
import json_reader
import numpy as np


def visualize_with_viewpoint(cloud, ok):
    # vis.add_geometry(cloud)
    # vis.poll_events()
    # vis.update_renderer()
    vis.add_geometry(ok)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.clear_geometries()


vis = o3d.visualization.Visualizer()
vis.create_window(width=500, height=500)
data_nu_scenes = data_nu_scenes.DataNuScenes()
json = json_reader.JsonReader()
max_lidar_points = 0
max_radar_points = 0

a = [[3.141592653589793, 3.141592653589793, 1.5707963267948966],
[0, 3.141592653589793, 3.141592653589793],
[0, 0, -1.5707963267948966]]

# For each pointcloud in dataset
# Check which scene, if same scene, get translation and add to cloud geometry
# If scene differ, start visualization
cur_scene = 'According to all known laws of aviation, there is no way a bee should be able to fly'
initial_translation = 0
o3d_pcd_lidar_comb = None
list_of_lidar = data_nu_scenes.list_of_files_lidar
start, end = json.get_scene(2, list_of_lidar)
print(f"Start {start} and end {end}")
print(f"DatasetSize: {data_nu_scenes.get_dataset_size()}")
rotation_possibilities = [0, math.pi, math.pi/2, -math.pi/2]
rotation_combinations = list(itertools.combinations_with_replacement(rotation_possibilities, 3))
all_comb = []
for comb in rotation_combinations:
    permutations = list(itertools.permutations(comb, 3))
    for huts in permutations:
        all_comb.append(huts)
unique_data = [list(x) for x in set(tuple(x) for x in all_comb)]
for comb in a:
    print(comb)
    # for f in range(start, end):
    #     o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
    #     translation, rotation = json.get_translation(data_nu_scenes.get_lidar_path(f))
    #     # rotation[3] += 1.5
    #     # rotation[0] += 1
    #     if initial_translation == 0:
    #         initial_translation = translation
    #     #translation = np.subtract(translation, initial_translation)
    #     translate = copy.deepcopy(o3d_pcd_lidar)
    #     translate.translate((translation[0], translation[1], translation[2]))
    #     rotate = copy.deepcopy(translate)
    #     R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    #     rotate.rotate(np.linalg.inv(R))
    #     deg_90 = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))
    #     rotate.rotate(deg_90)
    #     if o3d_pcd_lidar_comb is None:
    #         o3d_pcd_lidar_comb = rotate
    #     else:
    #         o3d_pcd_lidar_comb = o3d_pcd_lidar_comb + rotate
    #     # Radar
    #     o3d_pcd_radar_comb = o3d.geometry.PointCloud(data_nu_scenes.get_radar(f, max_radar_points)).paint_uniform_color(
    #         [0, 0, 0])
    joe = None
    for f in range(start, end):
        o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
        translation, rotation = json.get_translation(data_nu_scenes.get_lidar_path(f))
        # rotation[3] += 1.5
        # rotation[0] += 1
        if initial_translation == 0:
            initial_translation = translation
        #translation = np.subtract(translation, initial_translation)
        translate = copy.deepcopy(o3d_pcd_lidar)
        translate.translate((translation[0], translation[1], translation[2]))
        rotate = copy.deepcopy(translate)
        R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
        rotate.rotate(R)
        deg_90 = o3d.geometry.get_rotation_matrix_from_xyz((a[1][0], a[1][1], a[1][2]))
        rotate.rotate(deg_90)

        if joe is None:
            joe = rotate
        else:
            joe = joe + rotate
        # Radar
        o3d_pcd_radar_comb = o3d.geometry.PointCloud(data_nu_scenes.get_radar(f, max_radar_points)).paint_uniform_color(
            [0, 0, 0])
    visualize_with_viewpoint(o3d_pcd_lidar_comb, joe)
o3d.io.write_point_cloud("files/smallhuts.ply", o3d_pcd_lidar_comb)
gigahuts = 100


