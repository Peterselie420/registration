import copy
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

source = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part.ply"
target = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/full.ply"
offset = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part_offset.ply"
target_part = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/target_part.ply"
bunny_target = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/bunny/data/bun000.ply"
bunny_part = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/bunny/data/bun045.ply"


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


def apply_registration(eps):
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    source_points = reglib.load_data(bunny_part)
    target_points = reglib.load_data(bunny_target)
    pcd_offset = o3d.io.read_point_cloud(bunny_part).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    pcd_offset = o3d.io.read_point_cloud(bunny_target).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    # Run the registration algorithm
    start = time.time()
    trans = reglib.ndt(source=source_points, target=target_points, nr_iterations=eps, epsilon=1,
                       inlier_threshold=0.001, distance_threshold=0.05, downsample=0, visualize=False,
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


def render_scene_and_offset():
    nu_scenes_renderer = NuScenesRenderer()
    nu_scenes_renderer.__int__(nusc)
    scene = 0
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
    o3d.io.write_point_cloud(nu_scenes_globals.offset, part_cloud_transform)
    full_cloud = o3d.io.read_point_cloud(nu_scenes_globals.target).paint_uniform_color([0, 1, 0])
    part = o3d.io.read_point_cloud(nu_scenes_globals.target_part).paint_uniform_color([0, 0, 1])
    offset_cloud = o3d.io.read_point_cloud(nu_scenes_globals.offset).paint_uniform_color([1, 0, 0])
    nu_scenes_renderer.visualize_pcd([full_cloud, offset_cloud, part])


def center_cloud():
    full_cloud = o3d.io.read_point_cloud(nu_scenes_globals.target).paint_uniform_color([0, 1, 0])
    offset_cloud = o3d.io.read_point_cloud(nu_scenes_globals.offset).paint_uniform_color([1, 0, 0])
    mean, covariance = full_cloud.compute_mean_and_covariance()
    full_cloud.translate(-mean)
    offset_cloud.translate(-mean)
    o3d.io.write_point_cloud(nu_scenes_globals.target, full_cloud)
    o3d.io.write_point_cloud(nu_scenes_globals.offset, offset_cloud)


render_scene_and_offset()
center_cloud()
