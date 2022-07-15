from nuscenes import NuScenes
import open3d as o3d
import path_globals
from filter_machine import FilterMachine
from regression_machine import RegressionMachine
from render_machine import RenderMachine


def huts():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)

    scene = 0  # Select scene number
    random_sample = 20

    render_machine = RenderMachine(nusc, random_sample=random_sample)
    filter_machine = FilterMachine(nusc)
    regression_machine = RegressionMachine(nusc)

    # SCENE #
    pcd_lidar_scene, pcd_radar_scene = render_machine.render_scene(scene)
    render_machine.visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])

    # PART #
    pcd_lidar_part, pcd_radar_part, sample = render_machine.render_part(scene)
    render_machine.visualize_pcd([pcd_lidar_part, pcd_radar_part.paint_uniform_color([0, 0, 0])])

    # FILTER #
    pcd_lidar_scene_filtered = filter_machine.filter_scene(pcd_lidar_scene, scene)
    pcd_lidar_part_filtered = filter_machine.filter_cloud(pcd_lidar_part, sample)
    render_machine.visualize_pcd([pcd_lidar_scene_filtered.paint_uniform_color([0, 1, 0]),
                                  pcd_lidar_part_filtered.paint_uniform_color([1, 0, 0]),
                                  pcd_radar_part.paint_uniform_color([0, 0, 0])])

    # ZERO + CENTER #
    scene_mean, _ = pcd_radar_scene.compute_mean_and_covariance()
    pcd_lidar_scene_filtered_ZC = filter_machine.zero_and_center_cloud(pcd_lidar_scene_filtered, path_globals.lidar_scene_filter_ZC, scene_mean)
    pcd_radar_part_ZC = filter_machine.zero_and_center_cloud(pcd_radar_part, path_globals.radar_part_ZC, scene_mean)
    render_machine.visualize_pcd([pcd_lidar_scene_filtered_ZC.paint_uniform_color([0, 1, 0]), pcd_radar_part_ZC])

    pcd_big_filter = regression_machine.seek_linearity(pcd_lidar_scene_filtered_ZC)


if __name__ == '__main__':
    huts()
