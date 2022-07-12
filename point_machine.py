import os.path as osp
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion


class PointMachine:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def get_lidar_points(self, sample):
        points = self.get_points(sample, "LIDAR_TOP")
        return points

    def get_radar_points(self, sample):
        radar_list = ["RADAR_FRONT", "RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]
        points = np.empty((1, 18))
        for channel in radar_list:
            points = np.concatenate((points, self.get_points(sample, channel)))
        return points[1:]

    def get_points(self, sample, channel: str):
        pointsensor_token = sample['data'][channel]
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])

        if channel.__contains__("LIDAR"):
            pcd = LidarPointCloud.from_file(pcl_path)
        else:
            pcd = RadarPointCloud.from_file(pcl_path)

        # Transform from sensor frame to vehicle frame
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pcd.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pcd.translate(np.array(cs_record['translation']))

        # Transform from vehicle frame go global frame
        ego_record = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pcd.rotate(Quaternion(ego_record['rotation']).rotation_matrix)
        pcd.translate(np.array(ego_record['translation']))

        # Return transpose for own sanity
        return np.transpose(pcd.points)
