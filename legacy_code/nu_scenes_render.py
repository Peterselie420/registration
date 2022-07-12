import numpy as np
from nuscenes import NuScenes
import path_globals
from linear_regression import LinearRegression
from nu_scenes_helper import NuScenesHelper
import open3d as o3d


class NuScenesRenderer:
    def __int__(self, nusc: NuScenes):
        self.nusc = nusc
        self.nu_scenes_helper = NuScenesHelper()
        self.nu_scenes_helper.__int__(self.nusc)
        self.llr = LinearRegression()
        self.llr.__int__()

    @staticmethod
    def visualize_pcd(clouds):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=500, height=500)
        for cloud in clouds:
            vis.add_geometry(cloud)
            vis.poll_events()
            vis.update_renderer()
        vis.run()
        vis.destroy_window()

    def get_random_scene_sample(self, index: int):
        scene = self.nusc.scene[index]
        random = np.random.randint(0, scene['nbr_samples'])
        random = 20
        print(f"Randomly chosen sample: {random}")
        sample = scene['first_sample_token']
        i = 0
        while i < random:
            sample = self.nusc.get("sample", sample)
            sample = sample['next']
            i = i + 1
        return self.nusc.get("sample", sample)

    def render_scene(self, index: int, visualize: bool = False):
        assert index < self.nusc.scene.__len__(), f"Scene {index} outside of scope"
        scene = self.nusc.scene[index]
        current_sample_token = scene['first_sample_token']
        pcd_lidar_scene = np.empty((1, 4))
        pcd_radar_scene = np.empty((1, 18))
        mean_list = []
        while current_sample_token != "":
            sample = self.nusc.get("sample", current_sample_token)
            translation = self.nu_scenes_helper.get_vehicle_position(sample, "LIDAR_TOP")
            mean_list.append([translation[0], translation[1]])
            current_sample_token = sample['next']
        current_sample_token = scene['first_sample_token']
        pcd_lidar_scene = np.empty((1, 4))
        pcd_radar_scene = np.empty((1, 18))
        while current_sample_token != "":
            sample = self.nusc.get("sample", current_sample_token)
            pcd_lidar_scene = np.concatenate((pcd_lidar_scene, self.nu_scenes_helper.get_lidar_points(sample, mean_list=mean_list)), 0)
            pcd_radar_scene = np.concatenate((pcd_radar_scene, self.nu_scenes_helper.get_radar_points(sample)), 0)
            current_sample_token = sample['next']
        # pcd_lidar_scene = self.llr.zero_z_and_remove_dups(pcd_lidar_scene)
        pcd_lidar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_lidar_scene[:, [0, 1, 2]]))
        # Cut clouds to shape x-y-z and remove buffer point
        pcd_radar_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_scene[1:, [0, 1, 2]]))
        if visualize:
            self.visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])
        return pcd_lidar_scene, pcd_radar_scene

    def save_clouds(self, scene: int):
        pcd_lidar, pcd_radar = self.render_scene(scene, visualize=False)
        # pcd_lidar = subsample_cloud(15, pcd_lidar)
        o3d.io.write_point_cloud(nu_scenes_globals.target, pcd_lidar, write_ascii=False)
        sample = (self.get_random_scene_sample(scene))

        pcd_lidar_sample = self.nu_scenes_helper.get_lidar_points(sample, filter=False)[:, [0, 1, 2, 3]]
        # pcd_lidar_sample = self.llr.zero_z_and_remove_dups(pcd_lidar_sample)
        pcd_lidar_sample = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_lidar_sample[:, [0, 1, 2]]))
        o3d.io.write_point_cloud(nu_scenes_globals.target_part, pcd_lidar_sample, write_ascii=True)

        pcd_radar_sample = self.nu_scenes_helper.get_lidar_points(sample)[:, [0, 1, 2, 3]]
        # pcd_radar_sample = self.llr.zero_z_and_remove_dups(pcd_radar_sample)
        pcd_radar_sample = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_radar_sample[:, [0, 1, 2]]))
        o3d.io.write_point_cloud(nu_scenes_globals.source, pcd_radar_sample, write_ascii=True)
        print(len(pcd_radar_sample.points))