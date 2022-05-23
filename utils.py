import itertools
import os


def get_list_of_files(folder: str):
    list_of_files_radar_0 = sorted(
        filter(lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)))
    list_of_files_radar_0 = [(folder + '/{0}').format(it) for it in list_of_files_radar_0]
    return list_of_files_radar_0


def get_all_combinations_and_permutations(x, length: int):
    rotation_combinations = list(itertools.combinations_with_replacement(x, length))
    all_comb = []
    for comb in rotation_combinations:
        permutations = list(itertools.permutations(comb, 3))
        for huts in permutations:
            all_comb.append(huts)
    unique_data = [list(x) for x in set(tuple(x) for x in all_comb)]
    return unique_data
