import math
import open3d as o3d
import data_nu_scenes
import json_reader
import numpy as np
import pcd_render_utils


def visualize_with_viewpoint(clouds):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500)
    for cloud in clouds:
        vis.add_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
    vis.run()
    vis.clear_geometries()
    vis.destroy_window()


data_nu_scenes = data_nu_scenes.DataNuScenes()
json = json_reader.JsonReader()
max_lidar_points = 0
max_radar_points = 0

a = [[math.pi, math.pi, math.pi / 2], [0, 0, -math.pi / 2]]

initial_translation = 0
list_of_lidar = data_nu_scenes.list_of_files_lidar
start, end = json.get_scene(5, list_of_lidar)
print(f"Start {start} and end {end}")
print(f"DatasetSize: {data_nu_scenes.get_dataset_size()}")
full_cloud = None
radar_cloud = None
testcloud = None
for f in range(1, 2):
    o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
    o3d_pcd_radar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_radar(f, json, max_radar_points, False)))
    test = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(data_nu_scenes.get_radar(f, json, max_radar_points, True)))
    pcd_lidar_transform = pcd_render_utils.transform_cloud(f, json, data_nu_scenes, o3d_pcd_lidar).paint_uniform_color([0, 1, 0])
    pcd_radar_transform = pcd_render_utils.transform_cloud(f, json, data_nu_scenes, o3d_pcd_radar).paint_uniform_color([1, 0, 0])
    pcd_radar_test = pcd_render_utils.transform_cloud(f, json, data_nu_scenes, test).paint_uniform_color([0, 0, 1])

    if (full_cloud or radar_cloud) is None:
        full_cloud = pcd_lidar_transform
        radar_cloud = pcd_radar_transform
        testcloud = pcd_radar_test
    else:
        full_cloud = full_cloud + pcd_lidar_transform
        radar_cloud = radar_cloud + pcd_radar_transform
        testcloud = testcloud + pcd_radar_test
visualize_with_viewpoint([full_cloud, radar_cloud, testcloud])
visualize_with_viewpoint([full_cloud, testcloud])
#o3d.io.write_point_cloud("files/part.ply", full_cloud)
gigahuts = 100
