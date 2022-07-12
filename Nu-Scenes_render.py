from nuscenes import NuScenes

from filter_machine import FilterMachine
from render_machine import RenderMachine


def huts():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)

    scene = 0  # Select scene number
    random_sample = 20

    render_machine = RenderMachine(nusc, random_sample=random_sample)
    filter_machine = FilterMachine(nusc)

    pcd_lidar_scene, pcd_radar_scene = render_machine.render_scene(scene)
    render_machine.visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])

    pcd_lidar_part, pcd_radar_part, sample = render_machine.render_part(scene)
    render_machine.visualize_pcd([pcd_lidar_part, pcd_radar_part.paint_uniform_color([0, 0, 0])])

    pcd_lidar_scene_filtered = filter_machine.filter_scene(pcd_lidar_scene, scene)
    pcd_lidar_part_filtered = filter_machine.filter_cloud(pcd_lidar_part, sample)
    render_machine.visualize_pcd([pcd_lidar_scene_filtered, pcd_lidar_part_filtered.paint_uniform_color([0, 0, 0])])


if __name__ == '__main__':
    huts()
