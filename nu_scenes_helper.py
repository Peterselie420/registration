import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import os.path as osp
from pyquaternion import Quaternion


class NuScenesHelper:
    def __int__(self, nusc: NuScenes):
        self.nusc = nusc

    def get_lidar_points(self, sample):
        points = self.get_points(sample, "LIDAR_TOP", True)
        remove_indexes = []
        for i in range(0, len(points)):
            if points[i][3] < 20:
                remove_indexes.append(i)
        points = np.delete(points, remove_indexes, axis=0)
        return points

    def get_radar_points(self, sample):
        cloud = np.empty((1, 18))
        radar_list = ["RADAR_FRONT", "RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]
        for channel in radar_list:
            cloud = np.concatenate((cloud, self.get_points(sample, channel, False)))
        return cloud[1:]

    def get_points(self, sample, channel: str, is_lidar: bool):
        pointsensor_token = sample['data'][channel]
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        if is_lidar:
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
