import math
import numpy as np
from nuscenes import NuScenes
from open3d.cuda.pybind.geometry import PointCloud
import open3d as o3d
from scipy.stats import stats
from tqdm import tqdm
import path_globals


class RegressionMachine:

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.r_threshold = 0.99
        self.residual_threshold = 0.01
        self.buffer_size = 7
        self.middle = math.ceil(self.buffer_size / 2)

    @staticmethod
    def noise_filter(cloud: PointCloud):
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
        o3d.io.write_point_cloud(path_globals.radar_part_ZC_N, cloud, write_ascii=False)

    def perform_regression(self, buffer, index: int):
        x, y = [], []
        for point in buffer:
            x.append(point[0])
            y.append(point[1])
        assert len(x) == len(y), "No equal length between point arrays ;("
        if len(np.unique(x)) <= 1:
            #print("No unique points what the actual fuck")
            return []
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        if abs(r) < self.r_threshold:
            #print("Garbage relation, all points to the trash")
            return buffer
        prediction = slope * x[index] + intercept
        residual = abs(prediction - y[index])
        if residual > self.residual_threshold:
            #print("Point in question is outlier in current buffer")
            return [buffer[index]]
        return []

    def process_cloud(self, points):
        buffer = []
        remove_indexes = []
        print(len(points))
        with tqdm(total=(len(points) - self.buffer_size)) as pbar:
            for i in range(0, len(points)):
                if i >= len(points) - self.buffer_size:
                    break
                if len(buffer) < self.buffer_size:
                    buffer.append(points[i])
                if len(buffer) == self.buffer_size:
                    result = self.perform_regression(buffer, self.middle)
                    if len(result) == 1:
                        remove_indexes.append(i - self.middle)
                    if len(result) == self.buffer_size:
                        remove_indexes += [index + i for index in range(0, self.buffer_size)]
                    del buffer[0]
                pbar.update(1)
        unique_remove_indexes = np.unique(remove_indexes)
        if len(unique_remove_indexes) <= 1:
            return points
        print(len(points))
        points = np.delete(points, unique_remove_indexes, axis=0)
        print(len(points))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.lidar_scene_filter_ZC_LR, cloud, write_ascii=False)
        return cloud
