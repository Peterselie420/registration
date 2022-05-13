import os


def get_list_of_files(folder: str):
    list_of_files_radar_0 = sorted(
        filter(lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)))
    list_of_files_radar_0 = [(folder + '/{0}').format(it) for it in list_of_files_radar_0]
    return list_of_files_radar_0