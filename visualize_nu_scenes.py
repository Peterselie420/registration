import open3d as o3d

# Lidar
import data_nu_scenes


def visualize_with_viewpoint(cloud):
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()


data_nu_scenes = data_nu_scenes.DataNuScenes()
start = 0
max_lidar_points = 0
max_radar_points = 0

vis = o3d.visualization.Visualizer()
vis.create_window(width=500, height=500)

for f in range(start, 1):
    o3d_pcd_lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f, max_lidar_points))) \
        .paint_uniform_color([0, 0, 0])
    o3d_pcd_lidar_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f + 1, max_lidar_points))) \
        .paint_uniform_color([0, 0, 0])
    o3d_pcd_lidar_3 = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f + 2, max_lidar_points))) \
        .paint_uniform_color([0, 0, 0])
    o3d_pcd_lidar_4 = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f + 3, max_lidar_points))) \
        .paint_uniform_color([0, 0, 0])
    o3d_pcd_lidar_5 = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(data_nu_scenes.get_lidar(f + 4, max_lidar_points))) \
        .paint_uniform_color([0, 0, 0])
    # Radar
    o3d_pcd_radar_comb = o3d.geometry.PointCloud(data_nu_scenes.get_radar(f, max_radar_points)).paint_uniform_color(
        [0, 0, 0])
    visualize_with_viewpoint(o3d_pcd_lidar + o3d_pcd_lidar_2 + o3d_pcd_lidar_3 + o3d_pcd_lidar_4 + o3d_pcd_lidar_5)
    gigahuts = 100
