import os
import cv2
import glob
# import imutils
import numpy as np
# from skimage.rf_metrics import structural_similarity as compare_ssim
import torch

from ui.console_manager import ConsoleManager
from ui.json_dsl_manager import JsonDslManager
from utils.json_dataset_manager import JsonDatasetManager

# D
# cd C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\video_compression_nerfs\carla_rf
# conda activate carla_rf
# python vc_rfs.py --command "EVAL_RF_MODEL" --path "C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\video_compression_nerfs\carla_rf\scenarios\vc_rfs_scenario.json"

# P
# cd C:\Users\mDopiriak\Desktop\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\video_compression_rfs\carla_rf
# conda activate carla_rf
# python vc_rfs.py --command "ENCODE_IPFRAME_DATASET"
# python vc_rfs.py --command "PLOT_DEMO_DATA"
# python vc_rfs.py --command "EVAL_RF_MODEL"
# python vc_rfs.py --command "ENCODE_BATCH"
# python vc_rfs.py --command "EVAL_ENCODED_BATCH"
# python vc_rfs.py --command "PLOT_ENCODED_BATCH"
# python vc_rfs.py --command "SIMULATE_PACKET_LOSS"
# python vc_rfs.py --command "PLOT_PACKET_LOSS"
# python vc_rfs.py --command "CREATE_GT_MASK"
# python vc_rfs.py --command "EVAL_DOWNSTREAM_TASK"
# python vc_rfs.py --command "VIDEO_DEMO_DATA"


# 3 WEATHERS
# C:\Users\mDopiriak\Desktop\carla_city\encoded\test_batch_2024_11_12_12_07_24
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_22_00_01_42.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_22_01_41_07.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_22_09_21_42.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_22_11_18_00.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_15_28.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_22_44.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_32_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_39_53.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_44_49.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_50_37.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_01_55_47.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_02_13_18.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_02_19_10.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_02_27_11.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_02_36_41.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_02_43_09.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_11_27_43.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_11_36_48.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_11_42_46.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_11_47_35.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_11_56_49.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_03_01.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_07_42.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_17_36.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_23_55.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_32_49.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_40_09.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_45_03.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_50_30.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_52_58.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_12_56_52.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_00_51.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_04_27.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_13_23.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_18_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_23_22.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_32_47.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_37_02.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_41_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_46_38.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_51_55.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_13_51_55.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_14_07_05.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_14_11_28.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_14_23_57.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_14_30_21.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_14_45_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_15_20_38.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_15_28_18.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_15_35_32.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_15_45_10.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_16_06_58.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_16_11_27.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_16_22_34.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_16_38_16.csv"


# RAIN
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\encoded\\test_batch_2024_11_12_15_15_59_RAIN_VC"
# ACTUAL!!!!!!
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_37_55.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_46_07.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_54_33.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_02_49.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_08_36.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_25_28.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_35_37.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_57_45.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_08_13.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_23_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_29_22.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_47_00.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_06_32.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_16_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_23_44.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_35_20.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_45_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_01_06_37.csv"


# MAYBE OLDACTUAL
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_37_55.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_46_07.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_17_54_33.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_02_49.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_08_36.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_25_28.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_35_37.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_18_57_45.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_08_13.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_23_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_29_22.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_24_19_47_00.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_06_32.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_16_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_23_44.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_35_20.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_00_45_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_01_06_37.csv",
#NOT USED BELOW
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_25_01_27_22.csv"


#OLD
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_22_15_05_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_14_18_59.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_14_34_27.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_14_54_30.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_15_18_23.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_15_38_30.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_16_03_15.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_16_25_33.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_16_35_23.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_16_47_56.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_16_57_39.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_17_07_41.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_17_16_22.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_17_26_43.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_18_18_01.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_18_37_29.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_18_49_40.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_19_02_35.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_19_16_43.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_19_27_23.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_19_40_52.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_19_51_12.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_20_02_30.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_20_15_22.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_20_30_18.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_20_40_51.csv",
# "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\packet_loss_evaluation_2024_11_23_20_58_04.csv"



# 3WEATHERS
# "command": "PLOT_ENCODED_BATCH",
# "single_dataframe_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\encoded_batch_2024_11_12_14_34_10.csv",
# "plots_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\plots",
# "single_gt_dataframe_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\encoded_batch_2024_11_12_14_34_35.csv"

# RAIN AND FOG
# "command": "PLOT_ENCODED_BATCH",
# "single_dataframe_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\encoded_batch_2024_11_12_16_25_22.csv",
# "plots_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\plots",
# "single_gt_dataframe_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\dataframes\\encoded_batch_2024_11_12_16_26_03.csv"

# VIDEO DEMO DATA
# "command": "VIDEO_DEMO_DATA",
# "frames_datasets_paths": [
#     "C:\\Users\\mDopiriak\\Desktop\\carla_city\\a5\\splatfacto\\metrics\\GT_Carla",
#     "C:\\Users\\mDopiriak\\Desktop\\carla_city\\a5\\splatfacto\\metrics\\RF_Splatfacto"
# ],
# "plot_sequence_path": "C:\\Users\\mDopiriak\\Desktop\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\examples\\video_compression_rfs\\carla_rf\\datasets\\carla_ir_rl\\demo_data_vehicles_gs",
# "frames_path": "C:\\Users\\mDopiriak\\Desktop\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\examples\\video_compression_rfs\\carla_rf\\datasets\\carla_ir_rl\\demo_video_vehicles_gs\\frames",
# "video_path": "C:\\Users\\mDopiriak\\Desktop\\carla_city\\videos\\training_cav_rf_frames.mp4"


class VCRFs:
    def __init__(self):
        self.console_manager = ConsoleManager()
        self.args = self.console_manager.parse_args()
        self.json_dsl_manager = JsonDslManager()
        self.path = r"C:\Users\mDopiriak\Desktop\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\video_compression_rfs\carla_rf\scenarios/vc_rfs_scenario.json"

    def run(self):
        if self.args.command:
            self.json_dsl_manager.trigger_command(self.args.command, self.path)
    # def __init__(self):
    #     self.delta_detection_manager = DeltaDetectionManager()
    #     self.file_manager = FileManager()
    #     print(cv2.__version__)
    #     print(torch.cuda.is_available())
    #
    # def advanced_video_compression(self):
    #     dataset_path = rf"C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\video_compression_nerfs\carla_rf\datasets\carla_ir_rl"
    #     dataset_path_subfolders = self.file_manager.find_subfolders(dataset_path)
    #
    #     dataset_images_paths = []
    #     for dataset_path_subfolder in dataset_path_subfolders:
    #         print(dataset_path_subfolder)
    #         dataset_images_paths.append([dataset_images_path for dataset_images_path in
    #                                      glob.iglob(os.path.join(dataset_path_subfolder, "*.jpg"))])
    #
    #     # carla_train_images_paths, rf_nerfacto_images_paths = dataset_images_paths
    #
    #     detections = self.delta_detection_manager.detect_objects(dataset_images_paths)


if __name__ == "__main__":
    vcrfs = VCRFs()
    vcrfs.run()
