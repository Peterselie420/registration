import math
import numpy as np
from nuscenes import NuScenes
from open3d.cuda.pybind.geometry import PointCloud
import open3d as o3d
from tqdm import tqdm
import path_globals
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot


class RegressionMachine:

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    @staticmethod
    def seek_linearity(cloud: PointCloud):
        noise_reduction = []
        points = cloud.points
        count = 0
        with tqdm(total=(len(points))) as pbar:
            for i in range(0, len(points)):
                if i % 100 == 0:
                    print(len(noise_reduction))
                pbar.update(1)
                point = [points[i][0], points[i][1]]
                for j in range(0, len(points)):
                    if j == i:
                        continue
                    point_ = [points[j][0], points[j][1]]
                    if math.dist(point, point_) < 0.04:
                        count = count + 1
                    if count >= 15:
                        noise_reduction.append(points[i])
                        count = 0
                        break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(noise_reduction))
        o3d.io.write_point_cloud(path_globals.lidar_scene_filter_ZC_N, cloud, write_ascii=False)

    def get_vehicle_position(self, sample, channel: str = "LIDAR_TOP"):
        pointsensor_token = sample['data'][channel]
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        ego_record = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        translation = np.array(ego_record['translation'])
        return translation

    @staticmethod
    def zero_and_center_cloud(cloud: PointCloud, path: str, center_offset: [int, int, int]) -> PointCloud:
        points = cloud.points
        zero_points = []
        for point in points:
            zero_points.append([point[0], point[1], 0])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zero_points))
        cloud.translate([-center_offset[0], -center_offset[1], 0])
        o3d.io.write_point_cloud(path, cloud, write_ascii=False)
        return cloud

    def filter_cloud(self, pcd_lidar_part: PointCloud, sample: dict):
        print("Filter cloud\n")
        vehicle_position = self.get_vehicle_position(sample)
        cloud = self.apply_filter(pcd_lidar_part, [vehicle_position])
        o3d.io.write_point_cloud(path_globals.lidar_part_filter, cloud, write_ascii=False)
        return cloud

    def filter_scene(self, pcd_lidar_scene, scene: int):
        print("Filter scene\n")
        vehicle_positions = []
        scene = self.nusc.scene[scene]
        current_sample_token = scene['first_sample_token']
        while current_sample_token != "":
            sample = self.nusc.get("sample", current_sample_token)
            vehicle_positions.append(self.get_vehicle_position(sample))
            current_sample_token = sample['next']
        cloud = self.apply_filter(pcd_lidar_scene, vehicle_positions)
        o3d.io.write_point_cloud(path_globals.lidar_scene_filter, cloud, write_ascii=False)
        return cloud

    def apply_filter(self, cloud: PointCloud, vehicle_positions):
        points = cloud.points
        remove_indexes = []
        with tqdm(total=(len(points))) as pbar:
            for i in range(0, len(points)):
                pbar.update(1)
                x, y, z = points[i][0], points[i][1], points[i][2]
                if z <= self.height or z >= self.height_upper:
                    remove_indexes.append(i)
                    continue
                for position in vehicle_positions:
                    pos_x, pos_y = position[0], position[1]
                    if math.dist([x, y], [pos_x, pos_y]) < self.small_radius:
                        remove_indexes.append(i)
                        break
                    if math.dist([x, y], [pos_x, pos_y]) < self.radius and z < self.height_upper:
                        remove_indexes.append(i)
                        break
        points = np.delete(points, remove_indexes, axis=0)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
