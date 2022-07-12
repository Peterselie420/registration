import copy
import sys
import time
import numpy as np
import open3d as o3d
from numpy import ndarray
from nuscenes.nuscenes import NuScenes

# Load the nuScenes dataset (mini-split, in this case).
import path_globals
import reglib
from nu_scenes_render import NuScenesRenderer

nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)


def subsample_cloud(percentage: int, cloud: o3d.geometry.PointCloud):
    if percentage == 100:
        return cloud
    points = cloud.points
    size = len(points) - int(len(points) * (percentage / 100))
    points = np.delete(points, np.random.choice(len(points), size, replace=False), axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def generate_random_transformation_offset():
    x = np.random.randint(0, 4)
    y = np.random.randint(0, 4)
    z = np.random.randint(0, 4)
    x, y, z = 0.5, 0.7, 0
    rotation_list = []
    for i in range(0, 6):
        rotation_list.append(0.0)
    T = [[1, rotation_list[0], rotation_list[1], x],
         [rotation_list[2], 1, rotation_list[3], y],
         [rotation_list[4], rotation_list[5], 1, z],
         [0, 0, 0, 1]]
    print(T)
    return T, [x, y, z], rotation_list


def apply_registration():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    source_points = reglib.load_data(nu_scenes_globals.offset)
    target_points = reglib.load_data(nu_scenes_globals.target)
    pcd_offset = o3d.io.read_point_cloud(nu_scenes_globals.offset).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    pcd_offset = o3d.io.read_point_cloud(nu_scenes_globals.target).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    # Run the registration algorithm
    start = time.time()
    trans = reglib.ndt(source=source_points, target=target_points, nr_iterations=1, epsilon=1,
                       inlier_threshold=0.55, distance_threshold=5, downsample=0, visualize=False,
                       voxelize=True, resolution=1, step_size=0.1)
    print("Runtime:", time.time() - start)
    print(trans)
    return trans


def calculate_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> ndarray:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"
    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1) / len(
        weights)
    return np.sum(distances / weights)

rerender = True

nu_scenes_renderer = NuScenesRenderer()
nu_scenes_renderer.__int__(nusc)
scene = 0
if rerender:
     nu_scenes_renderer.save_clouds(scene)
# Get transformation matrix to offset part cloud
T_off, translation, rotation = generate_random_transformation_offset()
# Offset part cloud and save to ply format
part_cloud = o3d.io.read_point_cloud(nu_scenes_globals.source).paint_uniform_color([1, 0, 0])
part_cloud_copy = copy.deepcopy(part_cloud)
part_cloud_transform = part_cloud_copy.translate(translation)
cloud_center = part_cloud_transform.get_center()
part_cloud_transform = part_cloud_transform.rotate([[1.0, rotation[0], rotation[1]],
                                                    [rotation[2], 1.0, rotation[3]],
                                                    [rotation[4], rotation[5], 1.0]], center=cloud_center)
part_cloud_transform = part_cloud_transform.paint_uniform_color([0, 0, 0])
o3d.io.write_point_cloud(nu_scenes_globals.offset, part_cloud_transform, write_ascii=True)
# Show visualization for offset and original cloud with full cloud backdrop, so 3 total
full_cloud = o3d.io.read_point_cloud(nu_scenes_globals.target)
part_lidar_cloud = o3d.io.read_point_cloud(nu_scenes_globals.target_part).paint_uniform_color([0, 1, 1])
nu_scenes_renderer.visualize_pcd([part_cloud, part_cloud_transform, full_cloud, part_lidar_cloud])
h = 0
error = 999
while error > 0.1:
    # Apply NDT to the offset' cloud and get transformation matrix
    T_ndt = apply_registration()
    part_cloud_transform_copy = copy.deepcopy(part_cloud_transform)
    translation = [T_ndt[0, 3], T_ndt[1, 3], T_ndt[2, 3]]
    rotation = [[1, T_ndt[0, 1], T_ndt[0, 2]],
                [T_ndt[1, 0], 1, T_ndt[1, 2]],
                [T_ndt[2, 0], T_ndt[2, 1], 1]]
    print(translation)
    translation[2] = 0
    part_cloud_transform = o3d.io.read_point_cloud(nu_scenes_globals.offset)
    ndt_cloud = part_cloud_transform_copy.translate(translation).paint_uniform_color([0, 1, 0])
    cloud_center = ndt_cloud.get_center()
    ndt_cloud = ndt_cloud.rotate(rotation, center=cloud_center).paint_uniform_color([0, 1, 0])

    # Calculate NDT performance, maybe using the Translation matrix to calculate similarity
    error = calculate_error(part_cloud, ndt_cloud)
    print(error)
    mse = ((T_off - T_ndt)**2).mean(axis=None)
    print(mse)
    # Visualize NDT result with other clouds, so now 4 in total?

    o3d.io.write_point_cloud(offset, ndt_cloud, write_ascii=True)
    # Profit?!
    h = h + 1
nu_scenes_renderer.visualize_pcd([ndt_cloud, part_cloud, part_cloud_transform, part_lidar_cloud, full_cloud.paint_uniform_color([0, 0.4, 0.2])])
sys.exit()


source_points = reglib.load_data(offset)
target_points = reglib.load_data(target)

for epsilon in range(2, 50, 2):
    epsilon = epsilon / 10
    start = time.time()
    T_ndt = reglib.ndt(source=source_points, target=target_points, nr_iterations=50, epsilon=epsilon,
                       inlier_threshold=0.1, distance_threshold=2,
                       downsample=0, visualize=False, voxelize=True, resolution=8, step_size=0.3)
    print("Runtime:", time.time() - start)
    # Calculate NDT performance, maybe using the Translation matrix to calculate similarity

    translation = [T_ndt[0, 3], T_ndt[1, 3], T_ndt[2, 3]]
    rotation = [[1, T_ndt[0, 1], T_ndt[0, 2]],
                [T_ndt[1, 0], 1, T_ndt[1, 2]],
                [T_ndt[2, 0], T_ndt[2, 1], 1]]
    print(translation)
    translation[2] = 0
    part_cloud_transform_copy = copy.deepcopy(part_cloud_transform)
    ndt_cloud = part_cloud_transform_copy.translate(translation).paint_uniform_color([0, 1, 0])
    ndt_cloud = ndt_cloud.rotate(rotation, center=cloud_center).paint_uniform_color([0, 1, 0])

    error = calculate_error(part_cloud, ndt_cloud)
    print(error)
    mse = ((T_off - T_ndt) ** 2).mean(axis=None)
    print(mse)
    f = open("NDT_Result.txt", "a")
    f.writelines([f"Eps: {epsilon}, Inlier: {0.1}, Distance: {2},"
                  f"Stepsize: {0.3}, Error: {error}, MSE: {mse}\n"])
    f.close()
