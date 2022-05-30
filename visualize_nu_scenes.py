import copy
import math
import open3d as o3d
import data_nu_scenes
import json_reader
import numpy as np


def visualize_with_viewpoint(cloud):
    vis.add_geometry(cloud)
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

a = [[math.pi, math.pi, math.pi / 2], [0, 0, -math.pi / 2]]

initial_translation = 0
list_of_lidar = data_nu_scenes.list_of_files_lidar
start, end = json.get_scene(2, list_of_lidar)
print(f"Start {start} and end {end}")
print(f"DatasetSize: {data_nu_scenes.get_dataset_size()}")
full_cloud = None
for f in range(50, 51):
    o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
    translation, rotation = json.get_translation(data_nu_scenes.get_lidar_path(f))
    if initial_translation == 0:
        initial_translation = translation
    # translation = np.subtract(translation, initial_translation)
    # Apply translation provided by Nu_scenes
    translate = copy.deepcopy(o3d_pcd_lidar)
    translate.translate((translation[0], translation[1], translation[2]))
    # Apply rotation provided by Nu_scenes
    rotate = copy.deepcopy(translate)
    R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    rotate.rotate(R)
    # Apply rotation to fix offset
    deg_90 = o3d.geometry.get_rotation_matrix_from_xyz((a[1][0], a[1][1], a[1][2]))
    rotate.rotate(deg_90)

    if full_cloud is None:
        full_cloud = rotate
    else:
        full_cloud = full_cloud + rotate
    # Radar
    o3d_pcd_radar_comb = o3d.geometry.PointCloud(data_nu_scenes.get_radar(f, max_radar_points)).paint_uniform_color(
        [0, 0, 0])
visualize_with_viewpoint(full_cloud)
o3d.io.write_point_cloud("files/part.ply", full_cloud)
gigahuts = 100
