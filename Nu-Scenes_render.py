import math
import os
import glob

import numpy as np
from nuscenes import NuScenes
import open3d as o3d
import path_globals
import utils
from filter_machine import FilterMachine
from regression_machine import RegressionMachine
from render_machine import RenderMachine


def huts():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/nu_scenes', verbose=False)
    scene = 0  # Select scene number
    random_sample = 20
    max_concat = 20
    visualize = True
    nuke = True
    clear_all = True
    skip_part = None

    if nuke:
        utils.remove("/home/carpc/PycharmProjects/PythonRegistration/registration/clouds/")
    elif clear_all:
        concat_folder = utils.get_list_of_files(path_globals.scene_parts_raw)
        for c in concat_folder:
            os.remove(c)
        concat_folder = utils.get_list_of_files(path_globals.scene_parts_filter_ZC)
        for c in concat_folder:
            os.remove(c)
    for j in range(0, 10):
        scene = j
        utils.create_folders(scene)
        path_globals.scene = scene

        render_machine = RenderMachine(nusc, random_sample=random_sample)
        filter_machine = FilterMachine(nusc)
        regression_machine = RegressionMachine(nusc)

        # SCENE #
        pcd_lidar_scene, pcd_radar_scene = render_machine.render_scene(scene, render_parts=True, skip_part=skip_part)
        if visualize:
            render_machine.visualize_pcd([pcd_lidar_scene, pcd_radar_scene.paint_uniform_color([0, 0, 0])])

        # PART #
        pcd_lidar_part, pcd_radar_part, sample = render_machine.render_part(scene)
        if visualize:
            render_machine.visualize_pcd([pcd_lidar_part, pcd_radar_part.paint_uniform_color([0, 0, 0])])

        # FILTER #
        pcd_lidar_scene_filtered = filter_machine.filter_scene(pcd_lidar_scene, scene)
        pcd_lidar_part_filtered = filter_machine.filter_cloud(pcd_lidar_part, sample)
        if visualize:
            render_machine.visualize_pcd([pcd_lidar_scene_filtered.paint_uniform_color([0, 1, 0]),
                                          pcd_lidar_part_filtered.paint_uniform_color([1, 0, 0]),
                                          pcd_radar_part.paint_uniform_color([0, 0, 0])])

        # ZERO + CENTER #
        scene_mean, _ = pcd_radar_scene.compute_mean_and_covariance()
        pcd_lidar_scene_filtered_ZC = filter_machine.zero_and_center_cloud(pcd_lidar_scene_filtered,
                                                                           utils.split_cloud(path_globals.scene,
                                                                                             path_globals.lidar_scene_filter_ZC),
                                                                           scene_mean)
        filter_machine.zero_and_center_cloud(pcd_lidar_scene,
                                             utils.split_cloud(path_globals.scene, path_globals.lidar_scene_ZC),
                                             scene_mean)
        filter_machine.zero_and_center_cloud(pcd_radar_scene,
                                             utils.split_cloud(path_globals.scene, path_globals.radar_scene_ZC),
                                             scene_mean)
        pcd_radar_part_ZC = filter_machine.zero_and_center_cloud(pcd_radar_part, utils.split_cloud(path_globals.scene,
                                                                                                   path_globals.radar_part_ZC),
                                                                 scene_mean)
        if visualize:
            render_machine.visualize_pcd(
                [pcd_lidar_scene_filtered_ZC.paint_uniform_color([0, 1, 0]), pcd_radar_part_ZC])

        # FILTER AND ZERO AND CENTER ALL PART CLOUDS
        filter_machine.filter_center_zero_all(scene, scene_mean)

        # pcd_big_filter = regression_machine.noise_filter(pcd_radar_part_ZC)
        pcd_lidar_scene_LR = regression_machine.process_cloud(pcd_lidar_scene_filtered_ZC.points)
        for concat_lvl in range(0, max_concat):
            render_machine.render_concat_radar(concat_lvl)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_shit():
    files = utils.get_list_of_files(results_folder)
    for file in files:
        if file.__contains__("baseline_1_translation_2.5"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 10))
            fig.suptitle('Pokemon Stats by Generation')
            numbers = pd.read_csv(file).squeeze("columns")
            print(numbers.describe())
            sns.violinplot(ax=axes[0], data=numbers.iloc[:, 0])
            axes[0].set_ylim(0, 1)
            sns.violinplot(ax=axes[1], data=numbers.iloc[:, 1])
            plt.ylim(0, 1)
            sns.relplot(data=numbers)
            plt.ylim(0, 1)
            plt.show()


def plot_baseline(rotation: bool, tagg: str = None):
    tag = "translation"
    if rotation:
        tag = "_rotation"
    if tagg is not None:
        tag = tagg
    files = utils.get_list_of_files(results_folder)
    fig, axes = plt.subplots(2, 3, figsize=(6, 10))
    fig.suptitle('Lidar translation error')
    ax_index = 0
    for file in files:
        if file.__contains__(tag):
            print(file)
            numbers = pd.read_csv(file).squeeze("columns")
            print(numbers.describe())
            x_off = numbers.iloc[:, 0].to_numpy()
            y_off = numbers.iloc[:, 1].to_numpy()
            r_off = numbers.iloc[:, 2].to_numpy()
            offset = numbers.iloc[:, 3].to_numpy()
            rotation_offset = numbers.iloc[:, 4].to_numpy()
            absolute_off = []
            for i in range(0, len(x_off)):
                absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
            print(np.average(absolute_off))
            data = pd.DataFrame([absolute_off, numbers.iloc[:, 3]])
            print(ax_index / 3)
            print(ax_index % 2)
            if rotation:
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(rotation_offset, absolute_off)
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].title.set_text(file.split("results/")[1])
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].set(xlabel="Absolute offset",
                                                                             ylabel="Absolute error")
            else:
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(offset, absolute_off)
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].title.set_text(file.split("results/")[1])
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].set(xlabel="Absolute offset (m)",
                                                                             ylabel="Absolute error (m)")
            ax_index += 1
    ax_index = 0
    plt.show()


def plot_concat(rotation: bool, tagg: str = None):
    tag = "translation"
    if rotation:
        tag = "_rotation"
    if tagg is not None:
        tag = tagg
    files = utils.get_list_of_files(results_folder)
    fig, axes = plt.subplots(2, 3, figsize=(6, 10))
    fig.suptitle('Lidar error')
    ax_index = 0
    for file in files:
        if file.__contains__(tag):
            print(file)
            numbers = pd.read_csv(file).squeeze("columns")
            print(numbers.describe())
            x_off = numbers.iloc[:, 0].to_numpy()
            y_off = numbers.iloc[:, 1].to_numpy()
            r_off = numbers.iloc[:, 2].to_numpy()
            offset = numbers.iloc[:, 3].to_numpy()
            rotation_offset = numbers.iloc[:, 4].to_numpy()
            absolute_off = []
            for i in range(0, len(x_off)):
                absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
            print(np.average(absolute_off))
            data = pd.DataFrame([absolute_off, numbers.iloc[:, 3]])
            print(ax_index / 3)
            print(ax_index % 2)
            if rotation:
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(rotation_offset, r_off)
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].title.set_text(file.split("results/")[1])
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].set(xlabel="Absolute offset",
                                                                             ylabel="Absolute error")
            else:
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(offset, absolute_off)
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].title.set_text(file.split("results/")[1])
                axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].set(xlabel="Absolute offset",
                                                                             ylabel="Absolute error")
            ax_index += 1
    ax_index = 0
    plt.show()

def plot_concat_optim(tag: str):
    files = utils.get_list_of_files(folder)
    fig, axes = plt.subplots(2, 3, figsize=(6, 10))
    fig.suptitle('Radaer error')
    ax_index = 0
    for file in files:
        if file.__contains__(tag):
            print(file)
            numbers = pd.read_csv(file).squeeze("columns")
            print(numbers.describe())
            x_off = numbers.iloc[:, 0].to_numpy()
            y_off = numbers.iloc[:, 1].to_numpy()
            r_off = numbers.iloc[:, 2].to_numpy()
            x_off_ = numbers.iloc[:, 3].to_numpy()
            y_off_ = numbers.iloc[:, 4].to_numpy()
            r_off_ = numbers.iloc[:, 5].to_numpy()
            offset = numbers.iloc[:, 6].to_numpy()
            rotation_offset = numbers.iloc[:, 7].to_numpy()
            absolute_off = []
            for i in range(0, len(x_off)):
                absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
            absolute_off_ = []
            for i in range(0, len(x_off)):
                absolute_off_.append(math.sqrt(pow(x_off_[i], 2) + pow(y_off_[i], 2)))
            print(np.average(absolute_off_))
            print(ax_index / 3)
            print(ax_index % 2)
            axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(offset, absolute_off)
            axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].scatter(offset, absolute_off_)
            axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].title.set_text(file.split("build/")[1])
            axes[math.floor(ax_index / 3), math.floor(ax_index % 3)].set(xlabel="Absolute offset",
                                                                         ylabel="Absolute error")
            ax_index += 1
    ax_index = 0
    plt.show()


def print_mean_concat_performance(tag: str, samples: int = 1):
    files = utils.get_list_of_files(folder)
    for file in files:
        if file.__contains__(tag):
            numbers = pd.read_csv(file).squeeze("columns")
            for j in range(0, samples):
                # print(numbers.describe())
                x_off = numbers.iloc[:, 0 + 5 * j].to_numpy()
                y_off = numbers.iloc[:, 1 + 5 * j].to_numpy()
                r_off = numbers.iloc[:, 2 + 5 * j].to_numpy()
                iterations = numbers.iloc[:, 3 + 5 * j].to_numpy()
                elapsed_time = numbers.iloc[:, 4 + 5 * j].to_numpy()
                absolute_off = []
                for i in range(0, len(x_off)):
                    absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
                print(file)
                print(np.average(absolute_off))
                print(np.average(r_off))
                print(np.average(iterations))
                print(np.average(elapsed_time))

def print_mean_concat(tag: str, samples: int = 1):
    files = utils.get_list_of_files(folder)
    for file in files:
        if file.__contains__(tag):
            numbers = pd.read_csv(file).squeeze("columns")
            for j in range(0, samples):
                # print(numbers.describe())
                x_off = numbers.iloc[:, 0 + 3 * j].to_numpy()
                y_off = numbers.iloc[:, 1 + 3 * j].to_numpy()
                r_off = numbers.iloc[:, 2 + 3 * j].to_numpy()
                absolute_off = []
                for i in range(0, len(x_off)):
                    absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
                print(file)
                print(np.average(absolute_off))
                print(np.average(r_off))


def plot_random_offset(tag: str, samples: int = 1):
    files = utils.get_list_of_files(folder)
    for file in files:
        if file.__contains__(tag):
            numbers = pd.read_csv(file).squeeze("columns")
            for j in range(0, samples):
                # print(numbers.describe())
                x_off = numbers.iloc[:, 0 + 3 * j].to_numpy()
                y_off = numbers.iloc[:, 1 + 3 * j].to_numpy()
                r_off = numbers.iloc[:, 2 + 3 * j].to_numpy()
                offset = numbers.iloc[:, 7]
                print(abs(numbers.iloc[:, 7]).describe())
                sns.violinplot(data=offset)
                plt.ylim([-4, 4])
                plt.ylabel("Offset (deg)")
                plt.title("Randomly generated rotation offset from -4-4")
                plt.show()


def plot_single(tag: str):
    files = utils.get_list_of_files(results_folder)
    for file in files:
        if file.__contains__(tag):
            print(file)
            numbers = pd.read_csv(file).squeeze("columns")
            print(numbers.describe())
            x_off = numbers.iloc[:, 0].to_numpy()
            y_off = numbers.iloc[:, 1].to_numpy()
            r_off = numbers.iloc[:, 2].to_numpy()
            offset = numbers.iloc[:, 3].to_numpy()
            rotation_offset = numbers.iloc[:, 4].to_numpy()
            absolute_off = []
            for i in range(0, len(x_off)):
                absolute_off.append(math.sqrt(pow(x_off[i], 2) + pow(y_off[i], 2)))
            print(np.average(absolute_off))
            data = pd.DataFrame([absolute_off, numbers.iloc[:, 3]])
            plt.scatter(offset, absolute_off)
            plt.title("Lidar translation error")
            plt.ylabel("Absolute error (m)")
            plt.xlabel("Absolute offset (m)")
            ax_index = 0
            plt.show()


folder = "/home/carpc/Documents/PCL_NDT/build/"
results_folder = "/home/carpc/Documents/PCL_NDT/test_results/lidar_baseline/"

if __name__ == '__main__':
    tag = "_"
    #plot_random_offset(tag, samples=2)
    #plot_random_offset(tag, samples=2)
    #plot_single(tag)
    #plot_concat(rotation=False)
    plot_baseline(rotation=False, tagg=tag)
    #plot_concat_optim(tag)
