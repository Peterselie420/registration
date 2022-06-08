import copy
import time
import numpy as np
import data_nu_scenes
import json_reader
import pcd_render_utils
import reglib
import open3d as o3d
from numpy.random import randint

source = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part.ply"
target = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/full.ply"
offset = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part_offset.ply"


# Run code to interact with PCL and get transformation matrix
def get_transformation_matrix(source: str = offset, target: str = target):
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    source_points = reglib.load_data(source)
    target_points = reglib.load_data(target)

    # Run the registration algorithm
    start = time.time()
    trans = reglib.ndt(source=source_points, target=target_points, nr_iterations=5000, epsilon=0.5,
                       inlier_threshold=0.005, distance_threshold=5.0, downsample=0, visualize=False)
    # resolution=12.0, step_size=0.5, voxelize=0)
    print("Runtime:", time.time() - start)
    print(trans)
    return trans


def visualize_pcd(clouds):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500)
    for cloud in clouds:
        vis.add_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
    vis.run()
    vis.destroy_window()


def generate_random_transformation_offset():
    T = np.random.random([4, 4]) / 1000
    T[0][0], T[1][1], T[2][2] = 1, 1, 1
    # Correct last row
    T[3][0], T[3][1], T[3][2] = 0, 0, 0
    T[3][3] = 1
    print(T)
    return T


def render_clouds(json: json_reader.JsonReader, data_nu_scenes: data_nu_scenes.DataNuScenes, scene: int):
    lidar = False
    list_of_lidar = data_nu_scenes.list_of_files_lidar
    start, end = json.get_scene(scene, list_of_lidar)
    pcd_lidar = pcd_render_utils.construct_cloud(json, data_nu_scenes, start, end, lidar=True)
    o3d.io.write_point_cloud(target, pcd_lidar, write_ascii=True)
    single_scene = randint(start, end)
    pcd_lidar_part = pcd_render_utils.construct_cloud(json, data_nu_scenes, single_scene, single_scene + 1, lidar=lidar)
    o3d.io.write_point_cloud(source, pcd_lidar_part, write_ascii=True)


re_render_clouds = True  # Will re-render clouds, takes some minutes
scene = 3  # Scene to render
# Render clouds if necessary
if re_render_clouds:
    data_nu_scenes = data_nu_scenes.DataNuScenes()
    json = json_reader.JsonReader()
    render_clouds(json, data_nu_scenes, scene)
# Get transformation matrix to offset part cloud
T_off = generate_random_transformation_offset()
# Offset part cloud and save to ply format
part_cloud = o3d.io.read_point_cloud(source).paint_uniform_color([1, 0, 0])
part_cloud_copy = copy.deepcopy(part_cloud)
part_cloud_transform = part_cloud_copy.transform(T_off).paint_uniform_color([0, 0, 0])
o3d.io.write_point_cloud(offset, part_cloud_transform, write_ascii=True)
# Show visualization for offset and original cloud with full cloud backdrop, so 3 total
full_cloud = o3d.io.read_point_cloud(target)
visualize_pcd([part_cloud, part_cloud_transform, full_cloud])
# Apply NDT to the offset' cloud and get transformation matrix
ndt_T = get_transformation_matrix()
part_cloud_transform_copy = copy.deepcopy(part_cloud_transform)
ndt_cloud = part_cloud_transform_copy.transform(ndt_T).paint_uniform_color([0, 0, 1])
# Visualize NDT result with other clouds, so now 4 in total?
visualize_pcd([ndt_cloud, part_cloud, full_cloud, part_cloud_transform])
# Calculate NDT performance, maybe using the Translation matrix to calculate similarity
# Profit?!
