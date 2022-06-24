import copy
import os.path as osp
import time

import numpy as np
import open3d as o3d
from numpy import ndarray
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# Load the nuScenes dataset (mini-split, in this case).
import reglib

nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)

source = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part.ply"
target = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/full.ply"
offset = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/part_offset.ply"
target_part = "/home/carpc/PycharmProjects/PythonRegistration/registration/files/target_part.ply"


def visualize_pcd(clouds):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500)
    for cloud in clouds:
        vis.add_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
    vis.run()
    vis.destroy_window()


def get_lidar_points(sample):
    return get_points(sample, "LIDAR_TOP", True)


def subsample_cloud(percentage: int, cloud: o3d.geometry.PointCloud):
    if percentage == 100:
        return cloud
    points = cloud.points
    size = len(points) - int(len(points) * (percentage / 100))
    points = np.delete(points, np.random.choice(len(points), size, replace=False), axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def get_radar_points(sample):
    cloud = np.empty((1, 18))
    radar_list = ["RADAR_FRONT", "RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]
    for channel in radar_list:
        cloud = np.concatenate((cloud, get_points(sample, channel, False)))
    return cloud[1:]


def get_points(sample, channel: str, is_lidar: bool):
    pointsensor_token = sample['data'][channel]
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if is_lidar:
        pcd = LidarPointCloud.from_file(pcl_path)
    else:
        pcd = RadarPointCloud.from_file(pcl_path)
    # Transform from sensor frame to vehicle frame
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pcd.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pcd.translate(np.array(cs_record['translation']))
    # Transform from vehicle frame go global frame
    ego_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pcd.rotate(Quaternion(ego_record['rotation']).rotation_matrix)
    pcd.translate(np.array(ego_record['translation']))
    # Return transpose for own sanity
    return np.transpose(pcd.points)


def get_random_scene_sample(index: int):
    scene = nusc.scene[index]
    random = np.random.randint(0, scene['nbr_samples'])
    random = 20
    print(f"Randomly chosen sample: {random}")
    sample = scene['first_sample_token']
    i = 0
    while i < random:
        sample = nusc.get("sample", sample)
        sample = sample['next']
        i = i + 1
    return nusc.get("sample", sample)


def generate_random_transformation_offset():
    x = np.random.randint(0, 4)
    y = np.random.randint(0, 4)
    z = np.random.randint(0, 4)
    x, y, z = 1, 1.4, 0
    rotation_list = []
    for i in range(0, 6):
        rotation_list.append(0.0)
    T = [[1, rotation_list[0], rotation_list[1], x],
         [rotation_list[2], 1, rotation_list[3], y],
         [rotation_list[4], rotation_list[5], 1, z],
         [0, 0, 0, 1]]
    print(T)
    return T, [x, y, z], rotation_list


def render_scene(index: int, visualize: bool = False):
    assert index < nusc.scene.__len__(), f"Scene {index} outside of scope"
    scene = nusc.scene[index]
    current_sample_token = scene['first_sample_token']
    pcd_lidar_scene = np.empty((1, 4))
    pcd_radar_scene = np.empty((1, 18))
    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)
        pcd_lidar_scene = np.concatenate((pcd_lidar_scene, get_lidar_points(sample)), 0)
        pcd_radar_scene = np.concatenate((pcd_radar_scene, get_radar_points(sample)), 0)
        current_sample_token = sample['next']
    # Cut clouds to shape x-y-z and remove buffer point
    pcd_lidar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_lidar_scene[1:, [0, 1, 2]]))
    pcd_radar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_scene[1:, [0, 1, 2]]))
    if visualize:
        visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])
    return pcd_lidar_scene, pcd_radar_scene


def save_clouds(scene: int):
    pcd_lidar, pcd_radar = render_scene(scene, visualize=False)
    pcd_lidar = subsample_cloud(15, pcd_lidar)
    o3d.io.write_point_cloud(target, pcd_lidar, write_ascii=True)
    sample = (get_random_scene_sample(scene))
    pcd_lidar_sample = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(get_lidar_points(sample)[:, [0, 1, 2]])).paint_uniform_color([1, 0, 0])
    o3d.io.write_point_cloud(target_part, pcd_lidar_sample, write_ascii=True)
    pcd_radar_sample = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(get_lidar_points(sample)[:, [0, 1, 2]])).paint_uniform_color([0, 0, 0])
    o3d.io.write_point_cloud(source, pcd_radar_sample, write_ascii=True)
    print(len(pcd_radar_sample.points))


def apply_registration():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    source_points = reglib.load_data(offset)
    target_points = reglib.load_data(target)
    pcd_offset = o3d.io.read_point_cloud(offset).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    pcd_offset = o3d.io.read_point_cloud(target).paint_uniform_color([1, 0, 0])
    print(pcd_offset.compute_mean_and_covariance())
    # Run the registration algorithm
    start = time.time()
    trans = reglib.ndt(source=source_points, target=target_points, nr_iterations=80, epsilon=4.2,
                       inlier_threshold=0.25, distance_threshold=8, downsample=0, visualize=False,
                       voxelize=True, resolution=12.0, step_size=0.5)
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


scene = 0
save_clouds(scene)
# Get transformation matrix to offset part cloud
T_off, translation, rotation = generate_random_transformation_offset()
# Offset part cloud and save to ply format
part_cloud = o3d.io.read_point_cloud(source).paint_uniform_color([1, 0, 0])
part_cloud_copy = copy.deepcopy(part_cloud)
part_cloud_transform = part_cloud_copy.translate(translation)
cloud_center = part_cloud_transform.get_center()
part_cloud_transform = part_cloud_transform.rotate([[1.0, rotation[0], rotation[1]],
                                                    [rotation[2], 1.0, rotation[3]],
                                                    [rotation[4], rotation[5], 1.0]], center=cloud_center)
part_cloud_transform = part_cloud_transform.paint_uniform_color([0, 0, 0])
o3d.io.write_point_cloud(offset, part_cloud_transform, write_ascii=True)
# Show visualization for offset and original cloud with full cloud backdrop, so 3 total
full_cloud = o3d.io.read_point_cloud(target)
part_lidar_cloud = o3d.io.read_point_cloud(target_part).paint_uniform_color([0, 1, 1])
visualize_pcd([part_cloud, part_cloud_transform, full_cloud, part_lidar_cloud])
# Apply NDT to the offset' cloud and get transformation matrix
T_ndt = apply_registration()
part_cloud_transform_copy = copy.deepcopy(part_cloud_transform)
translation = [T_ndt[0, 3], T_ndt[1, 3], T_ndt[2, 3]]
rotation = [[1, T_ndt[0, 1], T_ndt[0, 2]],
            [T_ndt[1, 0], 1, T_ndt[1, 2]],
            [T_ndt[2, 0], T_ndt[2, 1], 1]]
print(translation)
translation[2] = 0
ndt_cloud = part_cloud_transform_copy.translate(translation).paint_uniform_color([0, 1, 0])
ndt_cloud = ndt_cloud.rotate(rotation, center=cloud_center).paint_uniform_color([0, 1, 0])

# Calculate NDT performance, maybe using the Translation matrix to calculate similarity
error = calculate_error(part_cloud, ndt_cloud)
print(error)
mse = ((T_off - T_ndt)**2).mean(axis=None)
print(mse)
# Visualize NDT result with other clouds, so now 4 in total?
visualize_pcd([ndt_cloud, part_cloud, part_cloud_transform, part_lidar_cloud])
# Profit?!
