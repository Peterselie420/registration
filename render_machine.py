import numpy as np
from nuscenes import NuScenes
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud

import path_globals
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

    def render_scene(self, scene: int, visualize: bool = False) -> [PointCloud, PointCloud]:
        """
        :param scene: Scene index to be rendered
        :param visualize: Whether to visualize result, default false
        :return: Lidar cloud of scene, Radar cloud of scene
        """
        assert scene < self.nusc.scene.__len__(), f"Scene {scene} outside of scope"
        scene = self.nusc.scene[scene]
        current_sample_token = scene['first_sample_token']
        pcd_lidar_scene = np.empty((1, 4))
        pcd_radar_scene = np.empty((1, 18))
        while current_sample_token != "":
            sample = self.nusc.get("sample", current_sample_token)
            pcd_lidar_scene = np.concatenate((pcd_lidar_scene, self.point_machine.get_lidar_points(sample)), 0)
            pcd_radar_scene = np.concatenate((pcd_radar_scene, self.point_machine.get_radar_points(sample)), 0)
            current_sample_token = sample['next']
        # Cut clouds to shape x-y-z and remove buffer point
        pcd_lidar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_lidar_scene[1:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.lidar_scene, pcd_lidar_scene, write_ascii=False)
        pcd_radar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_scene[1:, [0, 1, 2]]))
        o3d.io.write_point_cloud(path_globals.radar_scene, pcd_lidar_scene, write_ascii=False)
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
