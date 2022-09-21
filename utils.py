import os
import shutil

import path_globals


def get_list_of_files(folder: str):
    list_of_files_radar_0 = sorted(
        filter(lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)))
    list_of_files_radar_0 = [(folder + '/{0}').format(it) for it in list_of_files_radar_0]
    return list_of_files_radar_0


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    os.mkdir(path)


def split_cloud(scene: int, path: str):
    begin, end = path.split("clouds/")
    return begin + f"clouds/scene_{scene}/" + end


def create_folders(scene: int):
    os.makedirs(split_cloud(scene, path_globals.scene_parts_concat))
    os.makedirs(split_cloud(scene, path_globals.scene_parts_raw))
    os.makedirs(split_cloud(scene, path_globals.scene_parts_filter_ZC))
