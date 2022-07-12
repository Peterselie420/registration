import math
import numpy as np
from nuscenes import NuScenes
from open3d.cuda.pybind.geometry import PointCloud
import open3d as o3d

import path_globals


class FilterMachine:

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.height = 2.3
        self.radius = 20

    def get_vehicle_position(self, sample, channel: str = "LIDAR_TOP"):
        pointsensor_token = sample['data'][channel]
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        ego_record = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        translation = np.array(ego_record['translation'])
        return translation

    def filter_cloud(self, pcd_lidar_part: PointCloud, sample: dict):
        vehicle_position = self.get_vehicle_position(sample)
        cloud = self.apply_filter(pcd_lidar_part, [vehicle_position])
        o3d.io.write_point_cloud(path_globals.lidar_part_filter, cloud, write_ascii=False)
        return cloud

    def filter_scene(self, pcd_lidar_scene, scene: int):
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
        for i in range(0, len(points)):
            x, y, z = points[i][0], points[i][1], points[i][2]
            if z <= self.height:
                remove_indexes.append(i)
                continue
            for position in vehicle_positions:
                pos_x, pos_y = position[0], position[1]
                if math.dist([x, y], [pos_x, pos_y]) < self.radius:
                    remove_indexes.append(i)
                    break
        points = np.delete(points, remove_indexes, axis=0)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
