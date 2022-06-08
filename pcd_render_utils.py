import copy
import math
from data_nu_scenes import DataNuScenes
from json_reader import JsonReader
import open3d as o3d

a = [[math.pi, math.pi, math.pi / 2], [0, 0, -math.pi / 2]]


def transform_cloud(f: int, json: JsonReader, data_nu_scenes: DataNuScenes, cloud):
    translation, rotation = json.get_translation(data_nu_scenes.get_lidar_path(f))
    # Apply translation provided by Nu_scenes
    translate = copy.deepcopy(cloud)
    translate.translate((translation[0], translation[1], translation[2]))
    # Apply rotation provided by Nu_scenes
    rotate = copy.deepcopy(translate)
    R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    rotate.rotate(R)
    # Apply rotation to fix offset
    deg_90 = o3d.geometry.get_rotation_matrix_from_xyz((a[1][0], a[1][1], a[1][2]))
    rotate.rotate(deg_90)
    return rotate


def construct_cloud(json: JsonReader, data_nu_scenes: DataNuScenes, start: int, end: int, lidar: bool = True):
    max_lidar_points: int = 0
    full_cloud = None
    for f in range(start, end):
        if lidar:
            pcd_dataset = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points)))
        else:
            pcd_dataset = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(data_nu_scenes.get_radar(f, json, test=False)))
        pcd_transform = transform_cloud(f, json, data_nu_scenes,
                                        pcd_dataset).paint_uniform_color([0, 1, 0])
        if full_cloud is None:
            full_cloud = pcd_transform

        else:
            full_cloud = full_cloud + pcd_transform
    return full_cloud
