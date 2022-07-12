import os

import data_utils
import utils
from json_reader import JsonReader


class DataNuScenes:
    def __init__(self):
        # Root Nu-Scenes
        directory_nu_scenes = "/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/nu_scenes"
        # Lidar source
        directory_lidar = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/LIDAR_TOP'
        # Radar back left source
        directory_radar_back_left = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/RADAR_BACK_LEFT'
        # Radar back right source
        directory_radar_back_right = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/RADAR_BACK_RIGHT'
        # Radar front left source
        directory_radar_front_left = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/RADAR_FRONT_LEFT'
        # Radar back right source
        directory_radar_front_right = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/RADAR_FRONT_RIGHT'
        # Radar front source
        directory_radar_front = directory_nu_scenes + '/v1.0-trainval01_blobs/samples/RADAR_FRONT'

        self.list_of_files_lidar, self.list_of_files_radar_back_left, self.list_of_files_radar_back_right, \
        self.list_of_files_radar_front_left, self.list_of_files_radar_front_right, \
        self.list_of_files_radar_front = [], [], [], [], [], []
        for i in range(1, len(os.listdir(directory_nu_scenes)) + 1):
            self.list_of_files_lidar += utils.get_list_of_files(directory_lidar.replace("01", f'0{i.__str__()}'))
            self.list_of_files_radar_back_left += \
                utils.get_list_of_files(directory_radar_back_left.replace("01", f'0{i.__str__()}'))
            self.list_of_files_radar_back_right += \
                utils.get_list_of_files(directory_radar_back_right.replace("01", f'0{i.__str__()}'))
            self.list_of_files_radar_front_left += \
                utils.get_list_of_files(directory_radar_front_left.replace("01", f'0{i.__str__()}'))
            self.list_of_files_radar_front_right += \
                utils.get_list_of_files(directory_radar_front_right.replace("01", f'0{i.__str__()}'))
            self.list_of_files_radar_front += utils.get_list_of_files(
                directory_radar_front.replace("01", f'0{i.__str__()}'))

        assert len(self.list_of_files_radar_front) == len(self.list_of_files_lidar) == \
               len(self.list_of_files_radar_front_left) == len(self.list_of_files_radar_front_right) == \
               len(self.list_of_files_radar_back_right) == len(self.list_of_files_radar_back_left)

        self.dataset_size = len(self.list_of_files_lidar)

    def get_lidar_path(self, idx: int):
        return self.list_of_files_lidar[idx]

    def get_lidar(self, idx: int, json: JsonReader, max_points: int = 0, square=False):
        return data_utils.get_lidar_cloud(self.list_of_files_lidar[idx], json, max_points, square)

    def get_num_of_radar_points(self, idx: int, json_reader: JsonReader):
        return self.get_radar(idx, json_reader=json_reader, points=True)

    def get_radar(self, idx: int, json_reader: JsonReader, points: bool = False, test: bool = False, comb: bool = True):
        if comb:
            cloud, num_points = data_utils.get_radar_cloud(self.list_of_files_radar_front[idx],
                                                           self.list_of_files_radar_front_left[idx],
                                                           self.list_of_files_radar_front_right[idx],
                                                           self.list_of_files_radar_back_left[idx],
                                                           self.list_of_files_radar_back_right[idx], json_reader, test,
                                                           combined=comb)
            return num_points if points else cloud
        else:
            return data_utils.get_radar_cloud(self.list_of_files_radar_front[idx],
                                              self.list_of_files_radar_front_left[idx],
                                              self.list_of_files_radar_front_right[idx],
                                              self.list_of_files_radar_back_left[idx],
                                              self.list_of_files_radar_back_right[idx], json_reader, test,
                                              combined=comb)

    def get_dataset_size(self):
        return self.dataset_size

    def get_split_size(self, train_percentage, val_percentage, test_percentage):
        train_size = int(self.dataset_size * (train_percentage / 100))
        val_size = int(self.dataset_size * (val_percentage / 100))
        test_size = self.dataset_size - val_size - train_size
        return train_size, val_size, test_size
