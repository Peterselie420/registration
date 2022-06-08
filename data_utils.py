import open3d as o3d
import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt

from json_reader import JsonReader

set_rng = RandomState(420)


def remove_random_points(points, threshold: int):
    size = int(len(points) - threshold)
    if size < 1 or threshold <= 0:
        return points
    points = np.delete(points, set_rng.choice(len(points), size, replace=False), axis=0)
    return points


def pcd_to_rgb(pcd):
    # disp_norm = cv2.normalize(src=disp, dst= disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    _min = np.amin(pcd)
    _max = np.amax(pcd)
    # disp_norm = disp - _min * 255.0 / (_max - _min)
    disp_norm = (pcd - _min) * 255.0 / (_max - _min)
    disp_norm = np.uint8(disp_norm)
    plt.imshow(disp_norm, aspect='auto')
    plt.show()


# X, Y, Z, Intensity, Distance?
def get_lidar_cloud(path: str, max_points: int = 0, square: bool = False):
    distance_threshold = 20
    bin_pcd = np.fromfile(path, dtype=np.float32)
    points = bin_pcd.reshape((-1, 5))[:, [0, 1, 2, 3, 4]]
    filter_points = []
    for point in points:
        if point[4] > distance_threshold:
            filter_points.append([point[0], point[1], point[2]])
    filter_points = np.array(filter_points)
    points = points[:, [0, 1, 2]]
    return o3d.utility.Vector3dVector(points)


def get_radar_cloud(path_front: str, path_front_left: str, path_front_right: str,
                    path_back_left: str, path_back_right: str, json_reader: JsonReader, test: bool):
    o3d_pcd_radar_front = o3d.io.read_point_cloud(path_front)
    xyz_load = np.asarray(o3d_pcd_radar_front.points)
    xyz_load = xyz_load[:, [1, 0, 2]]
    xyz_load[:, [0]] *= -1
    o3d_pcd_radar_front.points = o3d.utility.Vector3dVector(xyz_load)
    o3d_pcd_radar_front = transform_radar(path_front, json_reader, o3d_pcd_radar_front, test)
    # Radar front left
    o3d_pcd_radar_front_left = o3d.io.read_point_cloud(path_front_left)
    xyz_load = np.asarray(o3d_pcd_radar_front_left.points)
    xyz_load[:, [0, 1]] *= -1
    o3d_pcd_radar_front_left.points = o3d.utility.Vector3dVector(xyz_load)
    o3d_pcd_radar_front_left = transform_radar(path_front_left, json_reader, o3d_pcd_radar_front_left, test)
    # Radar front right - In Lidar frame
    o3d_pcd_radar_front_right = o3d.io.read_point_cloud(path_front_right)
    o3d_pcd_radar_front_right = transform_radar(path_front_right, json_reader, o3d_pcd_radar_front_right, test)
    # Radar back left
    o3d_pcd_radar_back_left = o3d.io.read_point_cloud(path_back_left)
    xyz_load = np.asarray(o3d_pcd_radar_back_left.points)
    xyz_load = xyz_load[:, [1, 0, 2]]
    xyz_load[:, [1]] *= -1
    o3d_pcd_radar_back_left.points = o3d.utility.Vector3dVector(xyz_load)
    o3d_pcd_radar_back_left = transform_radar(path_back_left, json_reader, o3d_pcd_radar_back_left, test)
    # Radar back right
    o3d_pcd_radar_back_right = o3d.io.read_point_cloud(path_back_right)
    xyz_load = np.asarray(o3d_pcd_radar_back_right.points)
    xyz_load = xyz_load[:, [1, 0, 2]]
    xyz_load[:, [1]] *= -1
    o3d_pcd_radar_back_right.points = o3d.utility.Vector3dVector(xyz_load)
    o3d_pcd_radar_back_right = transform_radar(path_back_right, json_reader, o3d_pcd_radar_back_right, test)
    o3d_pcd_radar_comb = o3d_pcd_radar_back_right + o3d_pcd_radar_back_left + o3d_pcd_radar_front_left + \
                         o3d_pcd_radar_front_right + o3d_pcd_radar_front
    points = np.asarray(o3d_pcd_radar_comb.points)
    print(len(points))
    return o3d.utility.Vector3dVector(points), len(points)


def transform_radar(file_location: str, json_reader: JsonReader, cloud, test: bool = False):
    if test:
        return cloud
    translation, rotation = json_reader.get_calib_sensor_transformation(file_location)
    cloud.translate((translation[0], translation[1], translation[2]))
    R = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    cloud.rotate(R)
    return cloud
