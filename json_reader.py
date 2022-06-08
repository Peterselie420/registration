import json


class JsonReader:
    def __init__(self):
        self.cur_scene = 'According to all known laws of aviation, there is no way a bee should be able to fly'
        root_meta = '/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/v1.0-trainval_meta/v1.0-trainval/'
        ego_pose = root_meta + 'ego_pose.json'
        sample_data = root_meta + "sample_data.json"
        sample = root_meta + "sample.json"
        calib_sensor = root_meta + "calibrated_sensor.json"

        # Read JSON files
        print("Parsing json ego pose")
        with open(ego_pose, 'r') as fcc_file:
            self.ego_pose_read = json.load(fcc_file)
        print("Parsing json sample data")
        with open(sample_data, 'r') as fcc_file:
            self.sample_data_read = json.load(fcc_file)
        print("Parsing json sample")
        with open(sample, 'r') as fcc_file:
            self.sample_read = json.load(fcc_file)
        print("Parsing json calib_sensors")
        with open(calib_sensor, 'r') as fcc_file:
            self.calib_sensors = json.load(fcc_file)

    # Look for file name in sample data, and then look for corresponding ego pose to get translation big oof
    def get_translation(self, file_location: str):
        file_name = file_location.split(
            "/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/nu_scenes/v1.0-trainval01_blobs/"
        )[1]
        for sample_data in self.sample_data_read:
            if sample_data['filename'] == file_name:
                for sample in self.sample_read:
                    if sample['token'] == sample_data['sample_token']:
                        if self.cur_scene != sample['scene_token']:
                            self.cur_scene = sample['scene_token']
                            print("New scene detected!")
                            break
                # Get ego pose transformation
                for ego_pose in self.ego_pose_read:
                    if ego_pose['token'] == sample_data['ego_pose_token']:
                        return ego_pose['translation'], ego_pose['rotation']

    def get_calib_sensor_transformation(self, file_location: str):
        file_name = file_location.split(
            "/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/nu_scenes/v1.0-trainval01_blobs/"
        )[1]
        for sample_data in self.sample_data_read:
            if sample_data['filename'] == file_name:
                for calib_sensor in self.calib_sensors:
                    if calib_sensor['token'] == sample_data['calibrated_sensor_token']:
                        return calib_sensor['translation'], calib_sensor['rotation']
        print("Big fuckie wuckie")
        return 0, 0

    def get_scene(self, index: int, file_locations):
        start = 0
        end = 0
        f = 0
        i = 0
        cur_scene = "According to all known laws of aviation, there is no way a bee should be able to fly"
        # For file in file location, check scene, and iterate index for each new scene found
        for file in file_locations:
            file_location = file.split(
                "/home/carpc/PycharmProjects/Lidar_to_Radar/dataset/nu_scenes/v1.0-trainval01_blobs/"
            )[1]
            for huts in self.sample_data_read:
                if huts['filename'] == file_location:
                    for megahuts in self.sample_read:
                        if megahuts['token'] == huts['sample_token']:
                            if megahuts['scene_token'] != cur_scene:
                                cur_scene = megahuts['scene_token']
                                i = i + 1
                                start = end
                                end = f
                                if i >= index + 1:
                                    return start, end
                                print("New scene detected!")
            f = f + 1
        print("Scene not found ;(")
