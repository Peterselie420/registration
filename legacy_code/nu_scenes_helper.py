import math
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import os.path as osp
from pyquaternion import Quaternion
import open3d as o3d


class NuScenesHelper:
    def __int__(self, nusc: NuScenes):
        self.nusc = nusc

    def get_lidar_points(self, sample, filter: bool = True, mean_list = None):
        points = self.get_points(sample, "LIDAR_TOP", True)
        if filter:
            remove_indexes = []
            if mean_list is not None:
                for i in range(0, len(points)):
                    for mean in mean_list:
                        if math.dist([points[i][0], points[i][1]], [mean[0], mean[1]]) < 2 or -9999 <= points[i][2] <= 2.35\
                                or (math.dist([points[i][0], points[i][1]], [mean[0], mean[1]]) < 1 and points[i][2] > 2.30):
                            remove_indexes.append(i)
                            break
            else:
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, [0, 1, 2]]))
                mean, covariance = cloud.compute_mean_and_covariance()
                for i in range(0, len(points)):
                    if -9999 <= points[i][2] <= 0.6:
                        remove_indexes.append(i)
                for i in range(0, len(points)):
                    # todo use vehicle position instead of cloud mean (same for mean list, map)
                    if math.dist([points[i][0], points[i][1]], [mean[0], mean[1]]) < 20:
                        remove_indexes.append(i)
            points = np.delete(points, remove_indexes, axis=0)
        return points

    def get_vehicle_position(self, sample, channel: str):
        pointsensor_token = sample['data'][channel]
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        ego_record = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        translation = np.array(ego_record['translation'])
        return translation

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
