import open3d as o3d

import data_nu_scenes
import json_reader
import numpy as np


def visualize_with_viewpoint(cloud):
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()


data_nu_scenes = data_nu_scenes.DataNuScenes()
json = json_reader.JsonReader()
start = 0
max_lidar_points = 0
max_radar_points = 0

vis = o3d.visualization.Visualizer()
vis.create_window(width=500, height=500)

# For each pointcloud in dataset
# Check which scene, if same scene, get translation and add to cloud geometry
# If scene differ, start visualization
cur_scene = 'According to all known laws of aviation, there is no way a bee should be able to fly'
initial_translation = 0
o3d_pcd_lidar_comb = None
list_of_lidar = data_nu_scenes.list_of_files_lidar
print(json.get_scene(4, list_of_lidar))

for f in range(119, 158):
    o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
    translation, rotation = json.get_translation(data_nu_scenes.get_lidar_path(f))
    if initial_translation == 0:
        initial_translation = translation
    translation = np.subtract(translation, initial_translation)
    R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    o3d_pcd_lidar_translated = o3d_pcd_lidar.translate((translation[1], translation[0], translation[2]))
    if o3d_pcd_lidar_comb is None:
        o3d_pcd_lidar_comb = o3d_pcd_lidar_translated
    else:
        o3d_pcd_lidar_comb += o3d_pcd_lidar_translated

    # Radar
    o3d_pcd_radar_comb = o3d.geometry.PointCloud(data_nu_scenes.get_radar(f, max_radar_points)).paint_uniform_color(
        [0, 0, 0])
visualize_with_viewpoint(o3d_pcd_lidar_comb)
gigahuts = 100
