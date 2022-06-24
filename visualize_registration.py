import copy
import sys
import time
import numpy as np
from numpy import ndarray
import json_reader
import pcd_render_utils
import reglib
import open3d as o3d
from numpy.random import randint
import data_nu_scenes

source = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part.ply"
source_no_calib = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part_no_calib.ply"
target = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/full.ply"
offset = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part_offset.ply"
target_part = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/target_part.ply"

file = "hi-ha-fucking-huts.txt"


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
    pcd_lidar = pcd_render_utils.construct_cloud(json, data_nu_scenes, start, end, lidar=False)
    o3d.io.write_point_cloud(source, pcd_lidar, write_ascii=True)
    single_scene = randint(start, end)
    # pcd_lidar_part = pcd_render_utils.construct_cloud(json, data_nu_scenes, single_scene, single_scene + 1, lidar=True)
    # o3d.io.write_point_cloud(target_part, pcd_lidar_part, write_ascii=True)
    # pcd_radar = pcd_render_utils.construct_cloud(json, data_nu_scenes, single_scene, single_scene + 1, lidar=lidar)
    # #o3d.io.write_point_cloud(source, pcd_radar, write_ascii=True)
    # pcd_radar = pcd_render_utils.construct_cloud(json, data_nu_scenes, single_scene, single_scene + 1,
    #                                              lidar=lidar, test=True)
    # o3d.io.write_point_cloud(source_no_calib, pcd_radar, write_ascii=True)
    # Save index to file for use outside function and between runs
    with open(file, 'w') as f:
        f.write('%d' % single_scene)


def calculate_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> ndarray:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"

    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1) / len(
        weights)
    return np.sum(distances / weights)


check_radar = True
re_render_clouds = True  # Will re-render clouds, takes some time
scene = 2  # Scene to render
# Render clouds if necessary
if re_render_clouds:
    data = data_nu_scenes.DataNuScenes()
    json = json_reader.JsonReader()
    render_clouds(json, data, scene)
if check_radar:
    # Code to check radar offset
    full_cloud = o3d.io.read_point_cloud(target).paint_uniform_color([0, 1, 0])
    part_cloud = o3d.io.read_point_cloud(source).paint_uniform_color([1, 0, 0])
    with open(file) as file:
        scene = int(file.read())
    # for i in range(0, 1):
    #     front, front_right, front_left, back_right, back_left = data.get_radar(scene + i, json, test=False, comb=False)
    #     front = pcd_render_utils.transform_cloud(data.list_of_files_radar_front[scene], json, front).paint_uniform_color([0.168, 0.560, 0.952])
    #     front_right = pcd_render_utils.transform_cloud(data.list_of_files_radar_front_right[scene], json, front_right).paint_uniform_color([0.168, 0.262, 0.952])
    #     front_left = pcd_render_utils.transform_cloud(data.list_of_files_radar_front_left[scene], json, front_left).paint_uniform_color([0.168, 0.917, 0.952])
    #     back_right = pcd_render_utils.transform_cloud(data.list_of_files_radar_back_right[scene], json, back_right).paint_uniform_color([0.960, 0.698, 0])
    #     back_left = pcd_render_utils.transform_cloud(data.list_of_files_radar_back_left[scene], json, back_left).paint_uniform_color([0.960, 0.047, 0])
    visualize_pcd([full_cloud, part_cloud])
    sys.exit()

# Get transformation matrix to offset part cloud
T_off = generate_random_transformation_offset()
# Offset part cloud and save to ply format
part_cloud = o3d.io.read_point_cloud(source).paint_uniform_color([1, 0, 0])
part_cloud_no_calib = o3d.io.read_point_cloud(source_no_calib).paint_uniform_color([1, 1, 0])
part_cloud_copy = copy.deepcopy(part_cloud)
part_cloud_transform = part_cloud_copy.transform(T_off).paint_uniform_color([0, 0, 0])
o3d.io.write_point_cloud(offset, part_cloud_transform, write_ascii=True)
# Show visualization for offset and original cloud with full cloud backdrop, so 3 total
full_cloud = o3d.io.read_point_cloud(target)
part_lidar_cloud = o3d.io.read_point_cloud(target_part).paint_uniform_color([0, 1, 1])
visualize_pcd([part_cloud, part_cloud_transform, full_cloud, part_lidar_cloud, part_cloud_no_calib])
# Apply NDT to the offset' cloud and get transformation matrix
T_ndt = get_transformation_matrix()
part_cloud_transform_copy = copy.deepcopy(part_cloud_transform)
ndt_cloud = part_cloud_transform_copy.transform(T_ndt).paint_uniform_color([0, 0, 1])
# Visualize NDT result with other clouds, so now 4 in total?
visualize_pcd([ndt_cloud, part_cloud, full_cloud, part_cloud_transform, part_lidar_cloud])
# Calculate NDT performance, maybe using the Translation matrix to calculate similarity
error = calculate_error(part_cloud, ndt_cloud)
print(error)
mse = ((T_off - T_ndt)**2).mean(axis=None)
print(mse)
# Profit?!
