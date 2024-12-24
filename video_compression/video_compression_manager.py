import os
import re
import time

import cv2
import glob
import subprocess
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg

from utils.file_manager import FileManager
from utils.image_manager import ImageManager
from video_compression.encoding_modes import EncodingModes
from video_compression.delta_segmentation_manager import DeltaSegmentationManager


class VideoCompressionManager:
    def __init__(self):
        self.file_manager = FileManager()
        self.image_manager = ImageManager()
        self.delta_segmentation_manager = DeltaSegmentationManager()

    def extract_info(self, filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        run_cam = parts[0]  # e.g., 'run0cam1'
        return run_cam

    def encode_batch(self, command_data):
        encoded_batch_path = command_data["encoding"]["encoded_batch_path"]
        # codecs = command_data["encoding"]["codecs"]
        rf_variants = command_data["rf_variants"]

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        encoded_batch_full_path = self.create_encoded_batch_folder(encoded_batch_path, command_data, timestamp)

        for rf_variant in rf_variants:
            self.process_rf_variant(rf_variant, encoded_batch_full_path, command_data)

    def create_encoded_batch_folder(self, encoded_batch_path, command_data, timestamp):
        encoded_batch_name = command_data["encoding"]["encoded_batch_name"]
        encoded_batch_full_path = os.path.join(encoded_batch_path, f"{encoded_batch_name}_{timestamp}")
        self.file_manager.create_folder(encoded_batch_full_path)
        return encoded_batch_full_path

    def process_rf_variant(self, rf_variant, encoded_batch_full_path, command_data):
        rf_variant_name = rf_variant["rf_variant"]
        encoded_rf_variant_full_path = os.path.join(encoded_batch_full_path, rf_variant_name)
        self.file_manager.create_folder(encoded_rf_variant_full_path)

        for rf_model in rf_variant["rf_models"]:
            self.process_rf_model(rf_model, encoded_rf_variant_full_path, command_data)

    def process_rf_model(self, rf_model, encoded_rf_variant_full_path, command_data):
        rf_name = rf_model["rf_name"]
        encoded_area_full_path = os.path.join(encoded_rf_variant_full_path, rf_name)
        self.file_manager.create_folder(encoded_area_full_path)

        dense_traffics = rf_model["traffic"]["dense"]
        sparse_traffics = rf_model["traffic"]["sparse"]

        for dense_traffic in dense_traffics:
            dense_traffic_full_path = os.path.join(encoded_area_full_path, "dense")
            self.file_manager.create_folder(dense_traffic_full_path)
            self.process_traffic(dense_traffic, dense_traffic_full_path, rf_model, command_data)

        for sparse_traffic in sparse_traffics:
            sparse_traffic_full_path = os.path.join(encoded_area_full_path, "sparse")
            self.file_manager.create_folder(sparse_traffic_full_path)
            self.process_traffic(sparse_traffic, sparse_traffic_full_path, rf_model, command_data)

    def process_traffic(self, sd_traffic, encoded_area_full_path, rf_model, command_data):
        weathers = sd_traffic["weathers"]
        for weather in weathers:
            self.process_weather(weather, encoded_area_full_path, rf_model, command_data)

    def process_weather(self, weather, encoded_area_full_path, rf_model, command_data):
        weather_name = weather["weather_name"]
        encoded_weather_full_path = os.path.join(encoded_area_full_path, weather_name)
        self.file_manager.create_folder(encoded_weather_full_path)

        gt_jpg_files, rf_jpg_files = self.load_image_files(weather)
        if len(gt_jpg_files) != len(rf_jpg_files):
            print("Different length of GT and RF jpg files!")
            return

        gt_image_dict, rf_image_dict = self.group_images(gt_jpg_files, rf_jpg_files)

        self.encode_images(weather, gt_image_dict, rf_image_dict, encoded_weather_full_path, rf_model, command_data)

    def load_image_files(self, weather):
        gt_dataset_path = weather["gt_dataset_path"]
        rf_dataset_path = weather["rf_dataset_path"]
        gt_jpg_files = [jpgfile for jpgfile in glob.iglob(os.path.join(gt_dataset_path, "*.jpg"))]
        rf_jpg_files = [jpgfile for jpgfile in glob.iglob(os.path.join(rf_dataset_path, "*.jpg"))]
        return gt_jpg_files, rf_jpg_files

    def group_images(self, gt_jpg_files, rf_jpg_files):
        gt_image_dict = defaultdict(list)
        rf_image_dict = defaultdict(list)

        for gt_file in gt_jpg_files:
            gt_run_cam = self.extract_info(gt_file)
            gt_image_dict[gt_run_cam].append(gt_file)

        for i, rf_file in enumerate(rf_jpg_files):
            if i < len(gt_jpg_files):
                gt_run_cam = self.extract_info(gt_jpg_files[i])
                rf_image_dict[gt_run_cam].append(rf_file)

        return gt_image_dict, rf_image_dict

    def unroll_batch_size(self, images_length, batch_size):
        batches = []

        num_full_batches = images_length // batch_size
        images_left = images_length % batch_size

        # Add full batches
        batches.extend([batch_size] * num_full_batches)

        # Add the last smaller batch if any images are left
        if images_left > 0:
            batches.append(images_left)

        return batches

    def get_gt_mask_paths(self, gt_files_path, gt_mask_folder_path):
        gt_mask_paths = []
        for file_path in gt_files_path:
            filename = os.path.basename(file_path)
            gt_mask_path = os.path.join(gt_mask_folder_path, filename)
            gt_mask_paths.append(gt_mask_path)
        return gt_mask_paths

    def encode_images(self, weather, gt_image_dict, rf_image_dict, encoded_weather_full_path, rf_model, command_data):
        resolutions = command_data["encoding"]["resolutions"]
        encoding_modes = command_data["encoding"]["modes"]
        batch_sizes = []

        for encoding_mode in encoding_modes:
            if encoding_mode == EncodingModes.BVC.value:
                encoded_full_path = os.path.join(encoded_weather_full_path, f"bvc")
                batch_sizes = [2]

            elif encoding_mode == EncodingModes.RFDVC.value:
                batch_sizes = rf_model["batch_sizes"]
                encoded_full_path = os.path.join(encoded_weather_full_path, f"rfdvc")

            elif encoding_mode == EncodingModes.VC.value:
                batch_sizes = rf_model["batch_sizes"]
                encoded_full_path = os.path.join(encoded_weather_full_path, f"vc")

            self.file_manager.create_folder(encoded_full_path)

            for run_cam in gt_image_dict.keys():
                # if run_cam != "run3cam1":
                #     continue

                encoded_run_cam_full_path = os.path.join(encoded_full_path, run_cam)
                self.file_manager.create_folder(encoded_run_cam_full_path)

                gt_files = gt_image_dict[run_cam]
                # batch_sizes = [len(gt_files)]
                gt_images_cv2 = [cv2.imread(gt_path) for gt_path in gt_files]

                if encoding_mode == EncodingModes.VC.value:
                    delta_images_cv2 = gt_images_cv2
                else:
                    gt_mask = command_data["encoding"]["gt_mask"]
                    if gt_mask:
                        gt_mask_path = weather["gt_mask_path"]
                        # print("GT_MASK_PATH=", gt_mask_path)
                        # print(gt_files)
                        # print()
                        # print(len(gt_files))
                        gt_mask_files = self.get_gt_mask_paths(gt_files, gt_mask_path)
                        # print(gt_mask_files)
                        # print()
                        # print(len(gt_mask_files))
                        gt_mask_images_cv2 = [cv2.imread(gt_mask_path) for gt_mask_path in gt_mask_files]
                        delta_images_cv2 = self.delta_segmentation_manager.segment_objects_gt_mask(gt_images_cv2, gt_mask_images_cv2)
                        # save_path = r"C:\Users\mDopiriak\Desktop\carla_city\gt_mask_delta_frame.jpg"
                        #
                        # # Save the image as a .jpg
                        # cv2.imwrite(save_path, delta_images_cv2[0])
                        # exit(0)

                        # for delta_image_cv2 in delta_images_cv2:
                        #     cv2.imshow("DELTA IMAGE", delta_image_cv2)
                        #     cv2.waitKey(0)
                        #     exit(0)
                        #
                        #     black_pixel_percentage = self.delta_segmentation_manager.count_black_pixels_percentage(delta_image_cv2)
                        #     print("**********")
                        #     print("^RUN_CAM^=", run_cam)
                        #     print(f"BLACK PIXELS = {black_pixel_percentage}%")
                        #     print("**********")
                        # continue
                        # exit(0)
                    else:
                        rf_files = rf_image_dict[run_cam]
                        rf_images_cv2 = [cv2.imread(rf_path) for rf_path in rf_files]
                        self.delta_segmentation_manager.set_parallel_execution(False)
                        # delta_images_cv2 = self.delta_segmentation_manager.segment_objects(gt_images_cv2, rf_images_cv2)
                        delta_images_cv2 = self.delta_segmentation_manager.segment_objects_dl(gt_images_cv2, rf_images_cv2)
                        exit(0)
                        # save_path = r"C:\Users\mDopiriak\Desktop\carla_city\ds_delta_frame.jpg"
                        #
                        # # Save the image as a .jpg
                        # cv2.imwrite(save_path, delta_images_cv2[0])
                        # exit(0)
                        # black_pixels_percentages = []
                        # for delta_image_cv2 in delta_images_cv2:
                        #     cv2.imshow("DELTA FRAME", delta_image_cv2)
                        #     cv2.waitKey(0)
                        # exit(0)
                            # black_pixels_percentages.append(self.delta_segmentation_manager.count_black_pixels_percentage(delta_image_cv2))
                        # print("**********")
                        # print("^RUN_CAM^=", run_cam)
                        # print(f"BLACK PIXELS = {black_pixels_percentages}%")
                        # print(f"BLACK PIXELS AVG = {sum(black_pixels_percentages) / len(black_pixels_percentages)}%")
                        # print("**********")

                        # continue

                for batch_size in batch_sizes:
                    encoded_batch_size_full_path = os.path.join(encoded_run_cam_full_path,
                                                                f"batch_{batch_size}")
                    self.file_manager.create_folder(encoded_batch_size_full_path)
                    # unrolled_batch_sizes = self.unroll_batch_size(len(gt_files), batch_size)

                    # for index, unrolled_batch_size in enumerate(unrolled_batch_sizes, start=1):
                        # encoded_unrolled_batch_size_full_path = os.path.join(encoded_batch_size_full_path, f"unrolled_batch_{index}")
                        # self.file_manager.create_folder(encoded_unrolled_batch_size_full_path)

                    for resolution in resolutions:
                        encoded_resolution_full_path = os.path.join(encoded_batch_size_full_path,
                                                                    f"res_{resolution[0]}x{resolution[1]}")
                        self.file_manager.create_folder(encoded_resolution_full_path)

                        delta_image = delta_images_cv2[0]
                        height, width, _ = delta_image.shape
                        if width != resolution[0] or height != resolution[1]:
                            delta_images_cv2 = [cv2.resize(delta_single_image, (resolution[0], resolution[1])) for
                                                delta_single_image in delta_images_cv2]

                        codecs = command_data["encoding"]["codecs"]
                        for codec in codecs:
                            self.encode_with_codec(resolution, encoding_mode, codec, delta_images_cv2,
                                                   encoded_resolution_full_path,
                                                   batch_size)

    def encode_with_codec(self, resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path,
                          batch_size):
        codec_name = codec["codec"]
        if codec_name == "H.264":
            self.encode_h264(resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size)
        elif codec_name == "H.265":
            self.encode_h265(resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size)
        elif codec_name == "AV1":
            self.encode_av1(resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size)

    def encode_h264(self, resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size):
        encoded_h264_full_path = os.path.join(encoded_resolution_full_path, "h264")
        self.file_manager.create_folder(encoded_h264_full_path)

        tasks = []
        with ThreadPoolExecutor() as executor:
            for preset in codec["configurations"]["presets"]:
                for crf in codec["configurations"]["crfs"]:
                    encoded_videos_full_path = os.path.join(encoded_h264_full_path, f"preset_{preset}_crf_{crf}")
                    self.file_manager.create_folder(encoded_videos_full_path)

                    start_index = 0
                    delta_images_cv2_len = len(delta_images_cv2)
                    encoded_video_batch_index = 1

                    while start_index < delta_images_cv2_len:
                        end_index = min(start_index + batch_size, delta_images_cv2_len)
                        delta_images_cv2_batch = delta_images_cv2[start_index:end_index]

                        encoded_video_path = os.path.join(encoded_videos_full_path,
                                                          f"encoded_video_{encoded_video_batch_index}.h264")
                        ffmpeg_cmd = self.create_h264_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf,
                                                                encoding_mode)
                        tasks.append((encoded_video_batch_index,
                                      executor.submit(self.run_ffmpeg, ffmpeg_cmd, delta_images_cv2_batch)))

                        start_index = end_index
                        encoded_video_batch_index += 1
            for index, future in sorted(tasks, key=lambda x: x[0]):
                future.result()

    def encode_h265(self, resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size):
        encoded_h265_full_path = os.path.join(encoded_resolution_full_path, "h265")
        self.file_manager.create_folder(encoded_h265_full_path)

        tasks = []
        with ThreadPoolExecutor() as executor:
            for preset in codec["configurations"]["presets"]:
                for crf in codec["configurations"]["crfs"]:
                    encoded_videos_full_path = os.path.join(encoded_h265_full_path, f"preset_{preset}_crf_{crf}")
                    self.file_manager.create_folder(encoded_videos_full_path)

                    start_index = 0
                    delta_images_cv2_len = len(delta_images_cv2)
                    encoded_video_batch_index = 1

                    while start_index < delta_images_cv2_len:
                        end_index = min(start_index + batch_size, delta_images_cv2_len)
                        delta_images_cv2_batch = delta_images_cv2[start_index:end_index]

                        encoded_video_path = os.path.join(encoded_videos_full_path,
                                                          f"encoded_video_{encoded_video_batch_index}.h265")
                        ffmpeg_cmd = self.create_h265_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf,
                                                                encoding_mode)
                        tasks.append((encoded_video_batch_index,
                                      executor.submit(self.run_ffmpeg, ffmpeg_cmd, delta_images_cv2_batch)))

                        start_index = end_index
                        encoded_video_batch_index += 1
            for index, future in sorted(tasks, key=lambda x: x[0]):
                future.result()

    def encode_av1(self, resolution, encoding_mode, codec, delta_images_cv2, encoded_resolution_full_path, batch_size):
        encoded_av1_full_path = os.path.join(encoded_resolution_full_path, "av1")
        self.file_manager.create_folder(encoded_av1_full_path)
        # Actual encoding logic for AV1 here

        # tasks = []
        # with ThreadPoolExecutor() as executor:
        #     for preset in codec["configurations"]["presets"]:
        #         for crf in codec["configurations"]["crfs"]:
        #             encoded_videos_full_path = os.path.join(encoded_av1_full_path, f"preset_{preset}_crf_{crf}")
        #             self.file_manager.create_folder(encoded_videos_full_path)
        #
        #             start_index = 0
        #             delta_images_cv2_len = len(delta_images_cv2)
        #             encoded_video_batch_index = 1
        #
        #             while start_index < delta_images_cv2_len:
        #                 end_index = min(start_index + batch_size, delta_images_cv2_len)
        #                 delta_images_cv2_batch = delta_images_cv2[start_index:end_index]
        #
        #                 encoded_video_path = os.path.join(encoded_videos_full_path,
        #                                                   f"encoded_video_{encoded_video_batch_index}.webm")
        #                 ffmpeg_cmd = self.create_av1_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf,
        #                                                         encoding_mode)
        #                 tasks.append((encoded_video_batch_index,
        #                               executor.submit(self.run_ffmpeg, ffmpeg_cmd, delta_images_cv2_batch)))
        #
        #                 start_index = end_index
        #                 encoded_video_batch_index += 1
        #     for index, future in sorted(tasks, key=lambda x: x[0]):
        #         future.result()

    def create_h264_ffmpeg_command(self, resolution, encoded_video_path, batch_size, preset, crf, encoding_mode):
        if encoding_mode == EncodingModes.BVC.value:
            return self.create_bvc_h264_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf)
        elif encoding_mode == EncodingModes.RFDVC.value:
            return self.create_rfdvc_h264_ffmpeg_command(resolution, encoded_video_path, preset, crf)
        elif encoding_mode == EncodingModes.VC.value:
            return self.create_vc_h264_ffmpeg_command(resolution, encoded_video_path, preset, crf)

    def create_h265_ffmpeg_command(self, resolution, encoded_video_path, batch_size, preset, crf, encoding_mode):
        if encoding_mode == EncodingModes.BVC.value:
            return self.create_bvc_h265_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf)
        elif encoding_mode == EncodingModes.RFDVC.value:
            return self.create_rfdvc_h265_ffmpeg_command(resolution, encoded_video_path, preset, crf)
        elif encoding_mode == EncodingModes.VC.value:
            return self.create_vc_h265_ffmpeg_command(resolution, encoded_video_path, preset, crf)

    def create_av1_ffmpeg_command(self, resolution, encoded_video_path, batch_size, preset, crf, encoding_mode):
        # if encoding_mode == EncodingModes.BVC.value:
        #     return self.create_bvc_av1_ffmpeg_command(resolution, encoded_video_path, batch_size, preset, crf)
        if encoding_mode == EncodingModes.RFDVC.value or encoding_mode == EncodingModes.VC.value:
            return self.create_rfdvc_av1_ffmpeg_command(resolution, encoded_video_path, preset, crf)

    def run_ffmpeg(self, ffmpeg_cmd, delta_images_cv2_batch):
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for delta_image in delta_images_cv2_batch:
                process.stdin.write(delta_image.tobytes())
            process.stdin.flush()
            process.stdin.close()
            stdout, stderr = process.communicate()  # Highlighted Change: Using communicate for proper closure
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
        except Exception as e:
            process.kill()
            process.communicate()
            print(f"Exception occurred: {e}")
        finally:
            process.stdin.close()
            process.wait()
            print("Encoding...")

    def create_bvc_h264_ffmpeg_command(self, resolution, encoded_video_path, batch_size, preset, crf):
        return [
            'ffmpeg',
            '-y',
            '-framerate', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{resolution[0]}x{resolution[1]}',
            '-i', 'pipe:0',
            '-frames:v', str(batch_size),
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-r', '1',
            encoded_video_path
        ]

    def create_bvc_h265_ffmpeg_command(self, resolution, encoded_video_path, batch_size, preset, crf):
        return [
            'ffmpeg',
            '-y',
            '-framerate', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{resolution[0]}x{resolution[1]}',
            '-i', 'pipe:0',
            '-frames:v', str(batch_size),
            '-c:v', 'libx265',
            '-preset', preset,
            '-crf', str(crf),
            '-r', '1',
            encoded_video_path
        ]

    def create_rfdvc_h264_ffmpeg_command(self, resolution, encoded_video_path, preset, crf):
        return [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-framerate', '30',  # Assuming 30 FPS, you can change this as needed
            '-f', 'rawvideo',  # Input format is image sequence from pipe
            # '-vcodec', 'mjpeg',  # Input codec for image sequence
            '-pix_fmt', 'rgb24',  # Pixel format is RGB24
            '-s', f'{resolution[0]}x{resolution[1]}',  # Resolution of input images
            '-i', 'pipe:0',  # Read input from stdin
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', preset,  # Preset for encoding speed/compression ratio
            '-aq-strength', '2.0',
            # '-x264-params', 'psy=0',
            '-tune', 'zerolatency',
            '-crf', str(crf),  # Constant rate factor for quality
            '-g', '30',
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Set scale
            encoded_video_path  # Output file path
        ]

    def create_vc_h264_ffmpeg_command(self, resolution, encoded_video_path, preset, crf):
        return [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-framerate', '30',  # Assuming 30 FPS, you can change this as needed
            '-f', 'rawvideo',  # Input format is image sequence from pipe
            # '-vcodec', 'mjpeg',  # Input codec for image sequence
            '-pix_fmt', 'rgb24',  # Pixel format is RGB24
            '-s', f'{resolution[0]}x{resolution[1]}',  # Resolution of input images
            '-i', 'pipe:0',  # Read input from stdin
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', preset,  # Preset for encoding speed/compression ratio
            '-tune', 'zerolatency',
            '-crf', str(crf),  # Constant rate factor for quality
            '-g', '30',
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Set scale
            encoded_video_path  # Output file path
        ]

    def create_rfdvc_h265_ffmpeg_command(self, resolution, encoded_video_path, preset, crf):
        return [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-framerate', '30',  # Assuming 30 FPS, you can change this as needed
            '-f', 'rawvideo',  # Input format is image sequence from pipe
            # '-vcodec', 'mjpeg',  # Input codec for image sequence
            '-pix_fmt', 'rgb24',  # Pixel format is RGB24
            '-s', f'{resolution[0]}x{resolution[1]}',  # Resolution of input images
            '-i', 'pipe:0',  # Read input from stdin
            '-c:v', 'libx265',  # Use H.264 codec
            '-preset', preset,  # Preset for encoding speed/compression ratio
            # '-aq-strength', '3.0',
            # '-x265-params', 'psy-rd=0:psy-rdoq=0',
            '-tune', 'zerolatency',
            '-crf', str(crf),  # Constant rate factor for quality
            '-g', '30',
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Set scale
            encoded_video_path  # Output file path
        ]

    def create_vc_h265_ffmpeg_command(self, resolution, encoded_video_path, preset, crf):
        return [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-framerate', '30',  # Assuming 30 FPS, you can change this as needed
            '-f', 'rawvideo',  # Input format is image sequence from pipe
            # '-vcodec', 'mjpeg',  # Input codec for image sequence
            '-pix_fmt', 'rgb24',  # Pixel format is RGB24
            '-s', f'{resolution[0]}x{resolution[1]}',  # Resolution of input images
            '-i', 'pipe:0',  # Read input from stdin
            '-c:v', 'libx265',  # Use H.264 codec
            '-preset', preset,  # Preset for encoding speed/compression ratio
            '-tune', 'zerolatency',
            '-crf', str(crf),  # Constant rate factor for quality
            '-g', '30',
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Set scale
            encoded_video_path  # Output file path
        ]

    def create_rfdvc_av1_ffmpeg_command(self, resolution, encoded_video_path, preset, crf):
        print("AV1 FFMPEG!!")
        return [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', '30',  # Set input frame rate
            '-f', 'rawvideo',  # Input format is raw video
            '-pix_fmt', 'rgb24',  # Pixel format
            '-s', f'{resolution[0]}x{resolution[1]}',  # Input resolution
            '-i', 'pipe:0',  # Read input from standard input
            '-c:v', 'libaom-av1',  # Use AV1 codec
            '-preset', str(preset),  # Encoding speed (0-8), lower is slower but better quality
            '-crf', str(crf),  # Constant Rate Factor for quality
            # '-b:v', '0',  # Enable Constant Quality mode
            '-vf', f'scale={resolution[0]}:{resolution[1]}',  # Scaling filter
            encoded_video_path  # Output file path
        ]

    def encode_ipframe_dataset(self, command_data):
        ipframe_datasets_path = command_data["ipframe_datasets_path"]
        encoded_videos_path = command_data["encoded_videos_path"]
        # encoded_videos_folder_name = single_scenario["encoded_videos_folder_name"]
        # resolution = single_scenario["resolution"]
        presets = command_data["h264_cfg"]["presets"]
        crfs = command_data["h264_cfg"]["crfs"]
        # res_1920x1080_preset_veryfast_crf_28
        # encoded_videos_full_path = encoded_videos_path + "\\" + encoded_videos_folder_name
        ipframe_datasets_subfolders_paths = self.file_manager.find_subfolders(ipframe_datasets_path)

        for ipframe_dataset_subfolder_path in ipframe_datasets_subfolders_paths:
            for preset in presets:
                for crf in crfs:
                    jpgfiles = [jpgfile for jpgfile in
                                glob.iglob(os.path.join(ipframe_dataset_subfolder_path, "*.jpg"))]
                    jpgfiles_len = len(jpgfiles)
                    single_jpg_file = jpgfiles[0]
                    width, height = self.image_manager.get_resolution(single_jpg_file)
                    encoded_videos_full_path = encoded_videos_path + "\\" + f"res_{width}x{height}_preset_{preset}_crf_{crf}"
                    self.file_manager.create_folder(encoded_videos_full_path)
                    file_name = single_jpg_file.split("\\")[-1].split(".")[0] + ".jpg"
                    single_jpg_file_index = single_jpg_file.find("\\" + file_name)
                    new_path = single_jpg_file[:single_jpg_file_index]

                    file_path = new_path + "\\" + re.sub(r'\d+', '%3d', file_name)
                    order: int = 1
                    for i in range(jpgfiles_len):
                        encoded_video_path = encoded_videos_full_path + fr"\encoded_video{i + 1}.h264"

                        subprocess.run(
                            fr"ffmpeg -framerate 1 -start_number {order} -i {file_path} -frames:v 2 -c:v libx264 -preset {preset} -crf {crf} -r 1 {encoded_video_path}")

                        order = order + 2
