import os

import numpy as np
from nuscenes import NuScenes
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud

import path_globals
import utils
from point_machine import PointMachine


class RenderMachine:

    def __init__(self, nusc: NuScenes, random_sample: int = None):
        self.nusc = nusc
        self.point_machine = PointMachine(nusc)
        self.random_sample = random_sample

    @staticmethod
    def visualize_pcd(clouds):
        """
        :param clouds: List of PointCloud Objects
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=500, height=500)
        for cloud in clouds:
            vis.add_geometry(cloud)
            vis.poll_events()
            vis.update_renderer()
        vis.run()
        vis.destroy_window()

    def get_random_scene_sample(self, scene: int):
        """
        :param scene: Scene index
        :return: Random sample from within the scene
        """
        scene = self.nusc.scene[scene]
        random = np.random.randint(0, scene['nbr_samples'])
        assert self.random_sample < scene['nbr_samples'], "Manual 'random' sample outside of scene range"
        if self.random_sample is not None:
            random = self.random_sample
        print(f"Randomly chosen sample: {random}")
        sample = scene['first_sample_token']
        i = 0
        while i < random:
            sample = self.nusc.get("sample", sample)
            sample = sample['next']
            i = i + 1
        return self.nusc.get("sample", sample)

    @staticmethod
    def render_concat_radar(concat_level: int, clear_concat_folder: bool = True):
        if clear_concat_folder:
            concat_folder = utils.get_list_of_files(path_globals.scene_parts_concat)
            for c in concat_folder:
                os.remove(c)
        clouds = utils.get_list_of_files(path_globals.scene_parts_filter_ZC)
        for cloud_path in clouds:
            if cloud_path.__contains__("radar"):
                sample_ = cloud_path.split(path_globals.scene_parts_filter_ZC + "/radar_")
                sample_ = int(sample_[1].split(".pcd")[0])
                print(f"Sample currently processing: {sample_}")
                if sample_ < concat_level:
                    continue
                pcd_radar_scene = o3d.io.read_point_cloud(cloud_path).points
                concat_level_ = concat_level
                while concat_level_ > 0:
                    concat_level_ = concat_level_ - 1
                    for cloud_path_ in clouds:
                        if cloud_path_.__contains__("radar"):
                            sample__ = cloud_path_.split(path_globals.scene_parts_filter_ZC + "/radar_")
                            sample__ = int(sample__[1].split(".pcd")[0])
                            if sample__ == sample_ - concat_level_ - 1:
                                print(f"Concatenating sample {sample__}")
                                pcd_radar_scene = np.concatenate(
                                    (pcd_radar_scene, o3d.io.read_point_cloud(cloud_path_).points), 0)
                o3d.io.write_point_cloud(path_globals.scene_parts_concat + "radar_" + concat_level.__str__() +
                                         "_" + sample_.__str__() + ".pcd",
                                         o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_scene)),
                                         write_ascii=False)

    def render_scene(self, scene: int, render_parts: bool = False, visualize: bool = False) \
            -> [PointCloud, PointCloud]:
        """
        :param render_parts: Whether to render each part in the scene
        :param scene: Scene index to be rendered
        :param visualize: Whether to visualize result, default false
        :return: Lidar cloud of scene, Radar cloud of scene
        """

        pcd_lidar_scene = np.empty((1, 4))
        pcd_radar_scene = np.empty((1, 18))
        sample_index = 0
        hack = scene
        while scene <= hack:
            assert scene < self.nusc.scene.__len__(), f"Scene {scene} outside of scope"
            scene_ = self.nusc.scene[scene]
            current_sample_token = scene_['first_sample_token']
            while current_sample_token != "":
                sample = self.nusc.get("sample", current_sample_token)
                if render_parts:
                    lidar_points = self.point_machine.get_lidar_points(sample)
                    lidar_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_points[1:, [0, 1, 2]]))
                    o3d.io.write_point_cloud(path_globals.scene_parts_raw + "lidar_" + sample_index.__str__() + ".pcd",
                                             lidar_cloud, write_ascii=False)
                    radar_points = self.point_machine.get_radar_points(sample)
                    radar_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(radar_points[1:, [0, 1, 2]]))
                    o3d.io.write_point_cloud(path_globals.scene_parts_raw + "radar_" + sample_index.__str__() + ".pcd",
                                             radar_cloud, write_ascii=False)
                    sample_index = sample_index + 1

                pcd_lidar_scene = np.concatenate((pcd_lidar_scene, self.point_machine.get_lidar_points(sample)), 0)
                pcd_radar_scene = np.concatenate((pcd_radar_scene, self.point_machine.get_radar_points(sample)), 0)
                current_sample_token = sample['next']
            print(f"Scene {scene} processed!")
            scene = scene + 1
        # Cut clouds to shape x-y-z and remove buffer point
        pcd_lidar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_lidar_scene[1:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.lidar_scene, pcd_lidar_scene, write_ascii=False)
        pcd_radar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_scene[1:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.radar_scene, pcd_radar_scene, write_ascii=False)
        if visualize:
            self.visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])
        return pcd_lidar_scene, pcd_radar_scene

    def render_part(self, scene) -> [PointCloud, PointCloud, dict]:
        """
        :param scene: Scene to take random sample from and render
        :return: Lidar cloud of random sample, Radar cloud of random sample, Sample used
        """
        sample = self.get_random_scene_sample(scene)
        pcd_lidar_part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            self.point_machine.get_lidar_points(sample)[:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.lidar_part, pcd_lidar_part, write_ascii=False)
        pcd_radar_part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            self.point_machine.get_radar_points(sample)[:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.radar_part, pcd_radar_part, write_ascii=False)
        return pcd_lidar_part, pcd_radar_part, sample
