import os
import re
import cv2
import glob
import torch
import lpips
import subprocess
import pandas as pd
from datetime import datetime
import torchvision.transforms.functional as tf

from utils.rf_metrics.metrics_manager import MetricsManager
from utils.json_dataset_manager import JsonDatasetManager
from utils.file_manager import FileManager


def Clpips(img1, img2):
    global device,loss_fn
    loss_fn.cuda()
    img1 = lpips.im2tensor(img1)
    img2 = lpips.im2tensor(img2)
    img1 = img1.cuda()
    img2 = img2.cuda()
    dist01 = loss_fn.forward(img1, img2).mean().detach().cpu().tolist()
    return dist01


class EvaluationManager:
    def __init__(self):
        self.metrics_manager = MetricsManager()
        self.json_dataset_manager = JsonDatasetManager()
        self.file_manager = FileManager()

    def eval_encoded_batch(self, command_data):
        print("EVAL_ENCODED_BATCH")

        """
        Traverses the directory structure to collect data from encoded files
        and compiles it into a DataFrame with specified columns. Sorts the DataFrame
        based on Resolution from highest to lowest.

        Parameters:
        - command_data (dict): A dictionary containing configuration data, including 'encoded_batch_path' and 'dataframes_path'.

        Returns:
        - None
        """
        root_dir = command_data["encoded_batch_path"]
        # Initialize a list to store all data entries
        data_entries = []

        # Define the methods and approaches to traverse
        methods = ['nerfacto', 'splatfacto']
        approaches = ['rfdvc', 'vc']
        codecs = ['h264', 'av1', 'h265']

        # Traverse each method
        for method in methods:
            method_path = os.path.join(root_dir, method)
            if not os.path.isdir(method_path):
                print(f"Method directory not found: {method_path}")
                continue
            print(f"Processing method: {method}")

            # Traverse each area within the method
            for area in os.listdir(method_path):
                area_path = os.path.join(method_path, area)
                if not os.path.isdir(area_path):
                    print(f"  Area directory not found: {area_path}")
                    continue
                print(f"  Processing area: {area}")

                # Traverse each traffic condition within the area
                for traffic in os.listdir(area_path):
                    traffic_path = os.path.join(area_path, traffic)
                    if not os.path.isdir(traffic_path):
                        print(f"    Traffic directory not found: {traffic_path}")
                        continue
                    print(f"    Processing traffic condition: {traffic}")

                    # Traverse each weather condition within the traffic condition
                    for weather in os.listdir(traffic_path):
                        weather_path = os.path.join(traffic_path, weather)
                        if not os.path.isdir(weather_path):
                            print(f"      Weather directory not found: {weather_path}")
                            continue
                        print(f"      Processing weather condition: {weather}")

                        # Traverse each approach within the weather condition
                        for approach in approaches:
                            approach_path = os.path.join(weather_path, approach)
                            if not os.path.isdir(approach_path):
                                print(f"        Approach directory not found: {approach_path}")
                                continue
                            print(f"        Processing approach: {approach}")

                            # Traverse each run-camera combination within the approach
                            for run_camera in os.listdir(approach_path):
                                run_camera_path = os.path.join(approach_path, run_camera)
                                if not os.path.isdir(run_camera_path):
                                    print(f"          Run-Camera directory not found: {run_camera_path}")
                                    continue

                                # Extract Run and Camera information
                                if 'cam' not in run_camera:
                                    print(f"          Invalid Run-Camera name (missing 'cam'): {run_camera}")
                                    continue
                                run_part, cam_part = run_camera.split('cam', 1)
                                run = run_part.strip()  # e.g., 'run0'
                                camera = 'cam' + cam_part.strip()  # e.g., 'cam1'
                                print(f"          Processing Run: {run}, Camera: {camera}")

                                # Traverse each batch within the run-camera directory
                                for batch in os.listdir(run_camera_path):
                                    batch_path = os.path.join(run_camera_path, batch)
                                    if not os.path.isdir(batch_path):
                                        print(f"            Batch directory not found: {batch_path}")
                                        continue

                                    # Extract Batch_Size from batch directory name
                                    if not batch.startswith('batch_'):
                                        print(f"            Invalid batch name (missing 'batch_'): {batch}")
                                        continue
                                    try:
                                        batch_size = int(batch.split('_')[1])
                                    except (IndexError, ValueError):
                                        print(f"            Invalid batch size in batch name: {batch}")
                                        continue
                                    print(f"            Processing batch: {batch} (Batch_Size: {batch_size})")

                                    # Traverse each resolution within the batch
                                    for resolution in os.listdir(batch_path):
                                        res_dir = os.path.join(batch_path, resolution)
                                        if not os.path.isdir(res_dir):
                                            print(f"              Resolution directory not found: {res_dir}")
                                            continue

                                        # Traverse each codec within the resolution
                                        for codec in codecs:
                                            codec_path = os.path.join(res_dir, codec)
                                            if not os.path.isdir(codec_path):
                                                print(f"                Codec directory not found: {codec_path}")
                                                continue

                                            # Traverse each encoding parameter within the codec
                                            for encoding_param in os.listdir(codec_path):
                                                enc_param_dir = os.path.join(codec_path, encoding_param)
                                                if not os.path.isdir(enc_param_dir):
                                                    print(
                                                        f"                  Encoding Parameter directory not found: {enc_param_dir}")
                                                    continue

                                                # Traverse encoded files within the encoding parameter directory
                                                encoded_files = []
                                                for file in os.listdir(enc_param_dir):
                                                    file_path = os.path.join(enc_param_dir, file)
                                                    if os.path.isfile(file_path):
                                                        # Determine the codec from the file extension
                                                        file_ext = os.path.splitext(file)[1].lower()
                                                        if file_ext not in ['.h264', '.h265', '.av1']:
                                                            # Skip non-encoded files
                                                            continue

                                                        # Get file size in bytes
                                                        try:
                                                            file_size = os.path.getsize(file_path)
                                                        except OSError as e:
                                                            print(
                                                                f"                    Error accessing file {file_path}: {e}")
                                                            continue

                                                        encoded_files.append((file, file_size))

                                                if not encoded_files:
                                                    print(
                                                        f"                  No encoded files found in: {enc_param_dir}")
                                                    continue

                                                # Sort files alphabetically to assign Batch_Order
                                                encoded_files_sorted = sorted(encoded_files, key=lambda x: x[0])

                                                # Assign Batch_Order based on sorted order
                                                for idx, (file, size_bytes) in enumerate(encoded_files_sorted, start=1):
                                                    data_entries.append({
                                                        'Area': area,
                                                        'Traffic': traffic,
                                                        'Weather': weather,
                                                        'RF_Model': method,
                                                        'Approach': approach,
                                                        'Run': run,
                                                        'Camera': camera,
                                                        'Batch_Size': batch_size,
                                                        'Resolution': resolution.replace('res_', ''),
                                                        'Codec': codec,
                                                        'Encoding_Parameter': encoding_param,
                                                        'Batch_Order': idx,
                                                        'Total_Batch_Size_Bytes': size_bytes
                                                    })

        # Create the DataFrame
        df = pd.DataFrame(data_entries)

        # Function to calculate total pixels from resolution string
        def calculate_total_pixels(resolution):
            try:
                width, height = map(int, resolution.lower().split('x'))
                return width * height
            except:
                return 0  # Return 0 if resolution format is invalid

        # Apply the function to create a new column for total pixels
        df['Total_Pixels'] = df['Resolution'].apply(calculate_total_pixels)

        # Sort the DataFrame based on Total_Pixels in descending order (highest to lowest)
        df_sorted = df.sort_values(by='Total_Pixels', ascending=False).reset_index(drop=True)

        # Drop the auxiliary Total_Pixels column as it's no longer needed
        df_sorted = df_sorted.drop(columns=['Total_Pixels'])

        # Reorder columns as specified, with 'Area' first
        df_sorted = df_sorted[['Area', 'Traffic', 'Weather', 'RF_Model', 'Approach', 'Run', 'Camera', 'Batch_Size',
                               'Resolution', 'Codec', 'Encoding_Parameter', 'Batch_Order',
                               'Total_Batch_Size_Bytes']]

        # Display the sorted DataFrame
        print("\nCompiled and Sorted DataFrame:")
        print(df_sorted.head(20))  # Display first 20 rows for brevity

        # Save the DataFrame using the provided saving code
        try:
            # Save to CSV with timestamped filename
            dataframe_path = command_data["dataframes_path"]
            # Ensure the path ends with the OS-specific separator
            if not dataframe_path.endswith(os.sep):
                dataframe_path += os.sep
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            dataframe_filename = f"encoded_batch_{timestamp}.csv"
            dataframe_full_path = os.path.join(dataframe_path, dataframe_filename)
            df_sorted.to_csv(dataframe_full_path, index=False)
            print(f"\nDataFrame successfully saved to CSV: {dataframe_full_path}")

        except Exception as e:
            print(f"Error saving DataFrame: {e}")

    def eval_rf_model(self, command_data):
        global device, loss_fn
        print("EVAL RF MODEL")
        loss_fn = lpips.LPIPS(net='vgg', spatial=True)
        eval_dataset_path = command_data["eval_dataset_path"]
        # print("EVAL_DATASET_PATH=", eval_dataset_path)
        rf_variants = command_data["rf_variants"]
        # print("RF_VARIANTS=", rf_variants)
        file_format = "jpg"
        # data = {
        #         "rf_variants": [
        #             {
        #                 "rf_variant": "",
        #                 "rf_models": []
        #             },
        #             {
        #                 "rf_variant": "",
        #                 "rf_models": []
        #             }
        #         ]
        #     }

        data = {
                "rf_variants": []
            }

        for rf_variant in rf_variants:
            # print("rf_variant=", rf_variant)
            data_rf_variant = {
                "rf_variant": rf_variant["rf_variant"],
                "rf_models": []
            }
            rf_models = rf_variant["rf_models"]
            for rf_model in rf_models:
                rf_name = rf_model["rf_name"]
                gt_dataset_path = rf_model["gt_dataset_path"]
                rf_dataset_path = rf_model["rf_dataset_path"]
                rf_images = [cv2.imread(filename) for filename in
                               glob.iglob(os.path.join(rf_dataset_path, "*." + file_format))]
                reference_images = [cv2.imread(filename) for filename in
                                    glob.iglob(os.path.join(gt_dataset_path, "*." + file_format))]
                len_reference_images = len(reference_images)
                len_nerf_images = len(rf_images)

                if len_reference_images != len_nerf_images:
                    print("Datasets sizes are not equal!")
                    return

                # lpips_nerf_images = []
                # lpips_reference_images = []
                # for i in range(len_reference_images):
                #     lpips_nerf_images.append(tf.to_tensor(rf_images[i]).unsqueeze(0)[:, :3, :, :].cuda())
                #     lpips_reference_images.append(tf.to_tensor(reference_images[i]).unsqueeze(0)[:, :3, :, :].cuda())

                psnr_values = []
                ssim_values = []
                mse_values = []
                lpips_values = []

                for i, reference_image in enumerate(reference_images):
                    psnr = self.metrics_manager.psnr(reference_image, rf_images[i])
                    ssim = self.metrics_manager.ssim(reference_image, rf_images[i], multichannel=True, channel_axis=2)
                    mse = self.metrics_manager.mse(reference_image, rf_images[i])
                    # single_lpips = self.metrics_manager.lpips(reference_image, rf_images[i])
                    single_lpips = Clpips(rf_images[i], reference_image)
                    mse_values.append(mse)
                    psnr_values.append(round(psnr, 2))
                    ssim_values.append(round(ssim, 2))
                    lpips_values.append(round(single_lpips, 2))

                    print("\n*****")
                    print("ITERATION=", i)
                    print("MSE=", mse)
                    print("PSNR=", round(psnr, 2))
                    print("SSIM=", round(ssim, 2))
                    print("LPIPS=", round(single_lpips, 2))
                    print("*****\n")

                torch.cuda.empty_cache()
                data_rf_model = {
                    "rf_name": rf_name,
                    "psnr": psnr_values,
                    "ssim": ssim_values,
                    "lpips": lpips_values
                }
                data_rf_variant["rf_models"].append(data_rf_model)

                print("AVG MSE=", round(sum(mse_values) / len(mse_values), 2))
                print("AVG PSNR=", round(sum(psnr_values) / len(psnr_values), 2))
                print("AVG SSIM=", round(sum(ssim_values) / len(ssim_values), 2))
                print("AVG LPIPS=", round(sum(lpips_values) / len(lpips_values), 2))
                # print(eval_data)
            data["rf_variants"].append(data_rf_variant)
        self.json_dataset_manager.write_json_file(eval_dataset_path, data, "eval_models_data.json")

    def evaluate_encoded_ipframe(self, command_data):
        ipframe_encoded_videos_paths = command_data["ipframe_encoded_videos_path"]
        # pkt_size_file_name = single_scenario["pkt_size_file_name"]

        ipframe_encoded_videos_subfolders_paths = self.file_manager.find_subfolders(ipframe_encoded_videos_paths)
        # subfolder_name = self.file_manager.get_last_folder_name(ipframe_encoded_videos_subfolders_paths[0])
        # print("SUBFOLDER_NAME=", subfolder_name)
        # return
        for ipframe_encoded_videos_subfolder_path in ipframe_encoded_videos_subfolders_paths:
            h264files = [h264file for h264file in glob.iglob(os.path.join(ipframe_encoded_videos_subfolder_path, "*.h264"))]
            h264files_len = len(h264files)

            order: int = 1
            encoded_videos_data = []

            for i in range(h264files_len):
                file_path = re.sub(r"\d+(?=\.\w+$)", str(order), h264files[0])
                print(file_path)
                # filename = file_path.split("\\")[-1]

                subprocess_output = subprocess.check_output(
                    fr"ffprobe -show_frames {file_path}")

                order = order + 1

                subprocess_output_str = subprocess_output.decode('utf-8')
                subprocess_output_str_lines = subprocess_output_str.split('\n')
                packets = []
                for s in subprocess_output_str_lines:
                    if s.startswith("pkt_size"):
                        # Found a string that starts with "pkt_size"
                        number_str = s.replace("pkt_size=", "")
                        number = int(number_str)
                        packets.append(number)

                encoded_videos_data.append(packets)

            subfolder_name = self.file_manager.get_last_folder_name(ipframe_encoded_videos_subfolder_path)
            full_pkt_size_filename_path = ipframe_encoded_videos_subfolder_path + "\\" + f"pkt_size_{subfolder_name}.json"
            self.json_dataset_manager.write_file(full_pkt_size_filename_path, encoded_videos_data)
