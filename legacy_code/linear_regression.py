import math
import open3d as o3d
import numpy as np
from scipy import stats
from tqdm import tqdm


class LinearRegression:
    def __int__(self):
        # Threshold for performing linear regression, if bad fit, delete all points
        self.r_threshold = 0.995
        self.residual_threshold = 0.01
        self.buffer_size = 7
        self.middle = math.ceil(self.buffer_size / 2)

    def perform_regression(self, buffer, index: int):
        return []
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

    @staticmethod
    def zero_z_and_remove_dups(points):
        new_cloud = []
        for point in points:
            point[2] = 0
            new_cloud.append(point)
        new_cloud = np.asarray(new_cloud).reshape((-1, 4))
        # new_cloud = np.unique(new_cloud)
        return new_cloud

    def process_cloud(self, points):
        buffer = []
        remove_indexes = []
        print(len(points))
        points = self.zero_z_and_remove_dups(points)
        points = sorted(points, key=lambda x: x[3])
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
        return cloud
