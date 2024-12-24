import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing as mp
import ffmpeg  # Requires the `ffmpeg-python` package
from PIL import Image  # Requires the `Pillow` package
from functools import partial
import time
import traceback
import csv
import re
import uuid
import logging  # For logging
import tempfile  # For temporary file handling
import cv2
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# Set up logging
logging.basicConfig(
    filename='packet_loss_manager.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

def parse_run_camera(run_camera):
    """
    Parses the run_camera string to extract run and camera parts.

    Parameters:
    - run_camera (str): The run-camera string, e.g., "run0cam1"

    Returns:
    - tuple: (run, camera) if parsing is successful
    - (None, None) if parsing fails
    """
    pattern = r'^(run\d+)(cam\d+)$'
    match = re.match(pattern, run_camera, re.IGNORECASE)
    if match:
        run, camera = match.groups()
        return run.lower(), camera.lower()
    else:
        return None, None


def is_decodable(video_file):
    """
    Checks if the video file is decodable by ffprobe.

    Returns:
    - True if decodable, False otherwise.
    """
    try:
        ffmpeg.probe(video_file)
        return True
    except ffmpeg.Error:
        return False
    except Exception as e:
        logging.error(f"Unexpected error during decodability check: {e}")
        return False


def is_key_frame(nal_unit_type, codec='h264'):
    """
    Determines if the NAL unit type corresponds to a key frame.
    """
    if codec == 'h264':
        return nal_unit_type == 5  # IDR frame
    elif codec == 'h265':
        return nal_unit_type in [19, 20, 21]  # CRA, AP, IDR frames
    else:
        return False


def writer_function(queue, csv_file_path, header):
    """
    Function to write data entries to CSV file from a queue.
    """
    try:
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            # Write header only if file is empty
            if os.stat(csv_file_path).st_size == 0:
                writer.writeheader()
            while True:
                data_entry = queue.get()
                if data_entry == 'DONE':
                    break
                # Replace None with 'NaN' or appropriate placeholder
                for key, value in data_entry.items():
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        data_entry[key] = 'NaN'
                    elif value == np.inf:
                        data_entry[key] = 'Inf'
                    elif value == -np.inf:
                        data_entry[key] = '-Inf'
                writer.writerow(data_entry)
                csvfile.flush()
    except Exception as e:
        print(f"Error in writer_function: {e}")
        logging.error(f"Error in writer_function: {e}")
        traceback.print_exc()


def process_task(task_data, queue, lock, total_tasks, tasks_completed):
    """
    Processes a single file and returns the data entry.
    """
    # start_time = time.time()

    try:
        result = process_and_evaluate_file(
            file_path=task_data['file_path'],
            p_good_to_bad=task_data['p_good_to_bad'],
            p_bad_to_good=task_data['p_bad_to_good'],
            simulation_number=task_data['simulation_number'],
            corrupted_batch_path=task_data['corrupted_batch_path'],  # Pass the corrupted_batch_path
            encoded_batch_path=task_data['encoded_batch_path']  # To compute relative paths
        )

        if result is None:
            with lock:
                tasks_completed.value += 1
            return  # Skip if processing failed

        total_nal_units, lost_nal_units, packet_loss_rate, corrupted_file_size, bler_value, ssim_value = result

        # Collect data
        data_entry = {
            'Simulation_Number': task_data['simulation_number'],
            'Area': task_data['area'],
            'Traffic': task_data['traffic'],
            'Weather': task_data['weather'],
            'RF_Model': task_data['method'],
            'Approach': task_data['approach'],
            'Run': task_data['run'],
            'Camera': task_data['camera'],
            'Batch_Size': task_data['batch_size'],
            'Resolution': task_data['resolution'],
            'Codec': task_data['codec'],
            'Encoding_Parameter': task_data['encoding_param'],
            'File_Name': task_data['file'],
            'Original_File_Size_Bytes': os.path.getsize(task_data['file_path']),
            'Corrupted_File_Size_Bytes': corrupted_file_size,
            'Total_NAL_Units': total_nal_units,
            'Lost_NAL_Units': lost_nal_units,
            'Packet_Loss_Rate': packet_loss_rate,
            'BLER': bler_value,
            'SSIM': ssim_value
        }

        # end_time = time.time()
        # task_duration = end_time - start_time

        with lock:
            tasks_completed.value += 1
            # total_time.value += task_duration
            # avg_time_per_task = total_time.value / tasks_completed.value
            # remaining_tasks = total_tasks - tasks_completed.value
            # estimated_remaining_time = avg_time_per_task * remaining_tasks
            # estimated_total_time = avg_time_per_task * total_tasks
            # elapsed_time = total_time.value

            # Display progress with estimated times
            print(
                f"Processed ({tasks_completed.value}/{total_tasks}): "
                f"Area: {task_data['area']}, Traffic: {task_data['traffic']}, Weather: {task_data['weather']}, "
                f"Method: {task_data['method']}, Approach: {task_data['approach']}, Run: {task_data['run']} {task_data['camera']}, "
                f"Batch_Size: {task_data['batch_size']}, Resolution: {task_data['resolution']}, Codec: {task_data['codec']}, "
                f"Encoding_Param: {task_data['encoding_param']}, File: {task_data['file']}, "
                f"Simulation {task_data['simulation_number']}/{task_data['num_simulations']}\n"
                # f"Elapsed Time: {str(timedelta(seconds=int(elapsed_time)))}, "
                # f"Estimated Total Time: {str(timedelta(seconds=int(estimated_total_time)))}, "
                # f"Estimated Remaining Time: {str(timedelta(seconds=int(estimated_remaining_time)))}"
            )

        # Put data entry into the queue for the writer process
        queue.put(data_entry)

    except Exception as e:
        with lock:
            tasks_completed.value += 1
            print(f"Error processing file {task_data['file_path']}: {e}")
            logging.error(f"Error processing file {task_data['file_path']}: {e}")
            traceback.print_exc()
        return


def process_and_evaluate_file(file_path, p_good_to_bad, p_bad_to_good, simulation_number, corrupted_batch_path,
                              encoded_batch_path):
    """
    Processes the file with simulated packet loss at NAL unit level, reconstructs the corrupted file,
    decodes the original and corrupted files, and computes SNR and BER.

    Saves the corrupted file to the specified corrupted_batch_path only if simulation_number == 1,
    preserving folder structure.
    """
    try:
        # Initialize a new random number generator with a unique seed
        rng_seed = int(uuid.uuid4().int % 1_000_000_000)  # Use an integer modulus
        rng = np.random.default_rng(rng_seed)

        # Read the original file data
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Parse the file into NAL units
        nal_units = parse_nal_units(file_data)
        total_nal_units = len(nal_units)

        if total_nal_units == 0:
            print(f"No NAL units found in file: {file_path}")
            logging.warning(f"No NAL units found in file: {file_path}")
            return None

        # Determine the codec
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.h264':
            codec = 'h264'
        elif file_ext == '.h265':
            codec = 'h265'
        else:
            print(f"Unknown codec for file {file_path}")
            logging.warning(f"Unknown codec for file {file_path}")
            return None

        # Simulate packet loss using the Gilbert-Elliott model
        packet_loss_sequence = gilbert_elliott_model(p_good_to_bad, p_bad_to_good, total_nal_units, rng=rng)

        # # Adjust packet_loss_sequence to preserve critical NAL units and key frames
        # critical_nal_unit_types = []
        # if codec == 'h264':
        #     critical_nal_unit_types = [7, 8]  # SPS and PPS
        # elif codec == 'h265':
        #     critical_nal_unit_types = [32, 33, 34]  # VPS, SPS, PPS

        # # Adjust packet_loss_sequence to preserve critical NAL units and key frames
        # adjusted_packet_loss_sequence = []
        # for nal_unit, lost in zip(nal_units, packet_loss_sequence):
        #     nal_unit_type = get_nal_unit_type(nal_unit, codec=codec)
        #     if nal_unit_type is not None:
        #         if nal_unit_type in critical_nal_unit_types or is_key_frame(nal_unit_type, codec=codec):
        #             # if nal_unit_type in critical_nal_unit_types or is_key_frame(nal_unit_type, codec=codec):
        #             adjusted_packet_loss_sequence.append(0)  # Do not lose critical NAL units or key frames
        #         else:
        #             # Increase corruption probability for non-critical NAL units
        #             adjusted_packet_loss_sequence.append(rng.random() < 0.9)  # 90% chance of corruption
        #     else:
        #         adjusted_packet_loss_sequence.append(lost)  # Default to original corruption for unknown types

        # packet_loss_sequence = adjusted_packet_loss_sequence

        # Apply packet loss to the NAL units
        corrupted_nal_units = [start_code + nal_unit_data for (start_code, nal_unit_data), lost in
                               zip(nal_units, packet_loss_sequence) if not lost]
        lost_nal_units = sum(packet_loss_sequence)
        packet_loss_rate = lost_nal_units / total_nal_units if total_nal_units > 0 else 0

        # Reconstruct the corrupted data, including start codes
        corrupted_data = b''.join(corrupted_nal_units)

        # Determine if we should save the corrupted file
        save_corrupted = simulation_number == 1

        if save_corrupted:
            # Construct the corrupted file path within the corrupted_batch_path, preserving the folder structure
            relative_file_path = os.path.relpath(file_path, encoded_batch_path)
            corrupted_file_dir = os.path.join(corrupted_batch_path, os.path.dirname(relative_file_path))
            os.makedirs(corrupted_file_dir, exist_ok=True)

            # Ensure the corrupted file is named properly (include simulation number)
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            corrupted_file_name = f"{base_file_name}_corrupted_sim{simulation_number}{file_ext}"
            corrupted_file_path = os.path.join(corrupted_file_dir, corrupted_file_name)

            # Save the corrupted file
            with open(corrupted_file_path, "wb") as f:
                f.write(corrupted_data)

            # Get corrupted file size
            corrupted_file_size = len(corrupted_data)
        else:
            # For simulations beyond the first, use a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(corrupted_data)
            corrupted_file_size = len(corrupted_data)

        # Validate if corrupted file is decodable
        if save_corrupted:
            corrupted_file_to_check = corrupted_file_path
        else:
            corrupted_file_to_check = temp_file_path

        if not is_decodable(corrupted_file_to_check):
            print(f"Corrupted file is not decodable: {corrupted_file_to_check}")
            logging.warning(f"Corrupted file is not decodable: {corrupted_file_to_check}")
            # snr_value = None
            # ber_value = 1.0  # Assuming all bits lost
            ssim_value = 0.0
            bler_value = lost_nal_units / total_nal_units if total_nal_units > 0 else None
            # bler_value = 1.0
        else:
            # Decode original and corrupted videos and compute SNR and BER
            ssim_value = compute_quality_metrics(file_path, corrupted_file_to_check)
            # Compute BLER
            bler_value = lost_nal_units / total_nal_units if total_nal_units > 0 else None
            # ber_value = compute_ber(nal_units, packet_loss_sequence)

        if not save_corrupted and os.path.exists(temp_file_path):
            # Delete the temporary corrupted file
            os.remove(temp_file_path)

        return total_nal_units, lost_nal_units, packet_loss_rate, corrupted_file_size, bler_value, ssim_value
    except:
        print("ERROR")

def parse_nal_units(file_data):
    """
    Parses the video file data into a list of NAL units.

    Returns:
    - nal_units: A list of tuples (start_code, nal_unit_data).
    """
    nal_units = []
    i = 0
    length = len(file_data)

    while i < length:
        # Look for start code prefix (3 or 4 bytes)
        if file_data[i:i + 3] == b'\x00\x00\x01':
            start_code_length = 3
        elif file_data[i:i + 4] == b'\x00\x00\x00\x01':
            start_code_length = 4
        else:
            i += 1
            continue

        # Save the start code
        start_code = file_data[i:i + start_code_length]
        i += start_code_length
        nal_start = i  # Start of the NAL unit data (excluding start code)

        # Find the next start code
        while i < length:
            if file_data[i:i + 3] == b'\x00\x00\x01' or file_data[i:i + 4] == b'\x00\x00\x00\x01':
                break
            i += 1

        nal_end = i
        nal_unit_data = file_data[nal_start:nal_end]
        nal_units.append((start_code, nal_unit_data))

    return nal_units


def get_nal_unit_type(nal_unit, codec='h264'):
    """
    Returns the NAL unit type for H.264 or H.265 NAL unit.
    """
    start_code, nal_unit_data = nal_unit
    if codec == 'h264':
        if len(nal_unit_data) < 1:
            return None  # Not enough data
        nal_header = nal_unit_data[0]
        nal_unit_type = nal_header & 0x1F  # last 5 bits
        return nal_unit_type
    elif codec == 'h265':
        if len(nal_unit_data) < 2:
            return None  # Not enough data
        nal_header = nal_unit_data[:2]
        nal_unit_type = (nal_header[0] >> 1) & 0x3F  # bits 1-6 of the first byte
        return nal_unit_type
    else:
        return None


def add_black_regions(orig_array, corr_array):
    """
    Identifies black regions in orig_array and applies these regions to corr_array.

    Parameters:
    - orig_array: np.ndarray, original frame array with black regions (dtype=np.float64).
    - corr_array: np.ndarray, corrupted frame array to be updated (dtype=np.float64).

    Returns:
    - updated_corr_array: np.ndarray, updated corrupted frame with added black regions.
    """
    # Convert original array to uint8 and grayscale for contour detection
    orig_uint8 = np.clip(orig_array, 0, 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to identify black regions
    _, binary_mask = cv2.threshold(gray_orig, 1, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the black regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with black regions
    black_region_mask = np.zeros_like(binary_mask)
    cv2.drawContours(black_region_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply the black region mask to the corrupted frame
    updated_corr_array = corr_array.copy()
    updated_corr_array[black_region_mask == 255] = [0, 0, 0]  # Replace with black pixels (RGB)

    return updated_corr_array


def contains_rfdvc(path):
    """
    Checks if the given path contains the string 'rfdvc' in a case-insensitive manner.

    Parameters:
    - path: str, the path to check.

    Returns:
    - bool: True if 'rfdvc' is found in the path, otherwise False.
    """
    # Convert the path to lowercase and check for 'rfdvc'
    return 'rfdvc' in path.lower()

def compute_quality_metrics(original_file, corrupted_file):
    """
    Decodes the original and corrupted videos and computes SNR.

    Returns:
    - snr_value: SNR between original and corrupted video
    """
    try:
        # Decode original video frames
        original_frames = extract_frames(original_file)
        if not original_frames:
            print(f"Failed to extract frames from the original video: {original_file}")
            logging.warning(f"Failed to extract frames from the original video: {original_file}")
            return 0.0

        # Decode corrupted video frames
        corrupted_frames = extract_frames(corrupted_file)
        if not corrupted_frames:
            print(f"Failed to extract frames from the corrupted video: {corrupted_file}")
            logging.warning(f"Failed to extract frames from the corrupted video: {corrupted_file}")
            return 0.0

        # Ensure the same number of frames
        min_frames = min(len(original_frames), len(corrupted_frames))
        if min_frames == 0:
            print("No frames to compare.")
            logging.warning("No frames to compare.")
            return None

        original_frames = original_frames[:min_frames]
        corrupted_frames = corrupted_frames[:min_frames]

        # Compute average SNR
        # snr_values = []
        ssim_values = []
        # psnr_values = []
        index = 0
        corr_output_directory = r"C:\Users\mDopiriak\Desktop\carla_city\corrupted_frames"
        for orig_frame, corr_frame in zip(original_frames, corrupted_frames):
            # Convert frames to numpy arrays
            orig_array = np.array(orig_frame, dtype=np.float64)
            corr_array = np.array(corr_frame, dtype=np.float64)
            if contains_rfdvc(original_file) and contains_rfdvc(corrupted_file):
                corr_array = add_black_regions(orig_array, corr_array)
                print("*" * 40)
                print(orig_array.dtype, np.min(orig_array), np.max(orig_array))
                print(corr_array.dtype, np.min(corr_array), np.max(corr_array))
                print("*" * 40)
                print("*********************SAVING...")
                rfdvc_corr_output_path = os.path.join(corr_output_directory, f"rfdvc_corr_frame_{index}.png")
                cv2.imwrite(rfdvc_corr_output_path, corr_array)
                index = index + 1
            # orig_array = orig_array / 255.0
            # corr_array = corr_array / 255.0
            # print("*" * 40)
            # print(orig_array.dtype, np.min(orig_array), np.max(orig_array))
            # print(corr_array.dtype, np.min(corr_array), np.max(corr_array))
            # print("*" * 40)
            # print("*********************SAVING...")
            # output_path = os.path.join(corr_output_directory, f"corr_frame_{index}.png")
            # cv2.imwrite(output_path, corr_array)
            # index = index + 1

            # Check if frames have the same shape
            if orig_array.shape != corr_array.shape:
                print("Frame size mismatch, skipping frame.")
                logging.warning("Frame size mismatch, skipping frame.")
                continue

            # Compute SNR
            # snr = compute_snr(orig_array, corr_array)
            ssim = compute_ssim(orig_array, corr_array, multichannel=True, channel_axis=2, data_range=255)
            ssim_values.append(ssim)
            # psnr = compute_psnr(orig_array, corr_array)
            # print("$"*40)
            # print()
            # print("SSIM=", ssim)
            # print()
            # print("$"*40)
            # psnr_values.append(psnr)

        #     if snr is not None:
        #         snr_values.append(snr)
        #
        # if snr_values:
        #     snr_value = np.nanmean(snr_values)
        # else:
        #     snr_value = None
        #
        if ssim_values:
            ssim_value = np.nanmean(ssim_values)
        else:
            ssim_value = None

        # return ssim_value
        # if psnr_values:
        #     psnr_value = np.nanmean(psnr_values)
        # else:
        #     psnr_value = None

        return ssim_value

    except Exception as e:
        print(f"Error computing quality metrics: {e}")
        logging.error(f"Error computing quality metrics: {e}")
        traceback.print_exc()
        return None


def compute_snr(original_image, corrupted_image):
    """
    Computes the Signal-to-Noise Ratio (SNR) between the original and corrupted images.

    Returns:
    - SNR (dB) between the original and corrupted images.
    """
    original_image = np.array(original_image, dtype=np.float64)
    corrupted_image = np.array(corrupted_image, dtype=np.float64)

    # Calculate signal power (mean square of original image)
    signal_power = np.mean(original_image ** 2)

    # Calculate noise power (mean square error between original and corrupted images)
    noise_power = np.mean((original_image - corrupted_image) ** 2)

    # Handle edge cases for noise_power
    if noise_power == 0:
        return 100  # Assign a high finite SNR for perfect reconstruction
    if signal_power == 0:
        return -np.inf  # No signal, SNR is negative infinity

    return 10 * np.log10(signal_power / noise_power)


def compute_ber(nal_units, packet_loss_sequence):
    """
    Computes the Bit Error Rate (BER) based on lost NAL units.

    Returns:
    - BER: Bit Error Rate between original and corrupted data
    """
    total_bits = sum(len(nal_unit_data) * 8 for _, nal_unit_data in nal_units)
    lost_bits = sum(len(nal_unit_data) * 8 for (_, nal_unit_data), lost in zip(nal_units, packet_loss_sequence) if lost)

    ber = lost_bits / total_bits if total_bits > 0 else None
    return ber


def gilbert_elliott_model(p_good_to_bad, p_bad_to_good, total_units, rng=None):
    """
    Generates a packet loss sequence using the Gilbert-Elliott model.

    Parameters:
    - p_good_to_bad: Probability of transitioning from the Good state to the Bad state.
    - p_bad_to_good: Probability of transitioning from the Bad state to the Good state.
    - total_units: Total number of units (NAL units) to simulate.
    - rng: Random number generator (numpy.random.Generator). If None, a new generator is created.

    Returns:
    - packet_loss_sequence: A list where 1 represents a lost unit and 0 represents a successful unit.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure probabilities are valid
    if not (0 <= p_good_to_bad <= 1) or not (0 <= p_bad_to_good <= 1):
        raise ValueError("Transition probabilities must be between 0 and 1.")

    # Randomly choose initial state based on steady-state probabilities
    p_good = p_bad_to_good / (p_good_to_bad + p_bad_to_good) if (p_good_to_bad + p_bad_to_good) > 0 else 0
    state = 'good' if rng.random() < p_good else 'bad'
    packet_loss_sequence = []

    for _ in range(total_units):
        if state == 'good':
            packet_loss_sequence.append(0)  # No packet loss
            if rng.random() < p_good_to_bad:
                state = 'bad'
        else:
            packet_loss_sequence.append(1)  # Packet loss
            if rng.random() < p_bad_to_good:
                state = 'good'

    return packet_loss_sequence


def extract_frames(video_file):
    """
    Decodes the video file and extracts frames.

    Parameters:
    - video_file: The path to the video file from which frames will be extracted.

    Returns:
    - frames: A list of frames extracted from the video, where each frame is a PIL.Image object.
    """
    try:
        # Probe the video file to get metadata (like resolution)
        probe = ffmpeg.probe(video_file)
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if not video_streams:
            print(f"No video stream found in {video_file}")
            logging.warning(f"No video stream found in {video_file}")
            return []

        # Get video resolution
        width = int(video_streams[0]['width'])
        height = int(video_streams[0]['height'])

        # Use ffmpeg to decode the video and output raw frames in RGB format
        process = (
            ffmpeg
            .input(video_file, loglevel='error', err_detect='ignore_err')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args('-hide_banner')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        frame_size = width * height * 3
        frames = []

        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            if len(in_bytes) != frame_size:
                print("Incomplete frame read")
                logging.warning("Incomplete frame read")
                break
            # Convert raw bytes to an image frame
            frame = Image.frombytes('RGB', (width, height), in_bytes)
            frames.append(frame)

        # Read any remaining stderr to prevent deadlocks
        stderr = process.stderr.read()

        # Wait for the process to finish
        process.wait()

        # Check if the process completed successfully
        if process.returncode != 0:
            print(f"ffmpeg process returned non-zero exit code: {process.returncode}")
            logging.warning(f"ffmpeg process returned non-zero exit code: {process.returncode}")
            return []

        return frames

    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        logging.error(f"ffmpeg error: {e.stderr.decode()}")
        traceback.print_exc()
        return []

    except Exception as e:
        print(f"Error extracting frames from {video_file}: {e}")
        logging.error(f"Error extracting frames from {video_file}: {e}")
        traceback.print_exc()
        return []


class PacketLossManager:
    def __init__(self):
        pass

    def simulate_packet_loss(self, command_data):
        """
        Simulates packet loss on .h264 and .h265 files, reconstructs files after transmission,
        computes SNR, BER metrics, collects file size data, and saves the results to a CSV file.

        Parameters:
        - command_data (dict): A dictionary containing configuration data, including:
            - 'encoded_batch_path': Root directory path to start traversal.
            - 'dataframes_path': Directory path to save the resulting CSV file.
            - 'corrupted_batch_path': Directory path to save the corrupted files.
            - 'p_good_to_bad': Probability of transitioning from Good to Bad state.
            - 'p_bad_to_good': Probability of transitioning from Bad to Good state.
            - 'num_simulations': Number of simulations to run.
            - 'num_workers': Number of worker processes to use.

        Returns:
        - None
        """
        print("SIMULATE PACKET LOSS")
        dataframes_path = command_data.get("dataframes_path", "")
        encoded_batch_path = command_data.get("encoded_batch_path", "")
        corrupted_batch_path = command_data.get("corrupted_batch_path", "")  # New parameter
        p_good_to_bad = command_data.get("p_good_to_bad", 0.15)  # 0.15 0.05
        p_bad_to_good = command_data.get("p_bad_to_good", 0.25)  # 0.25 0.5
        num_simulations = command_data.get("num_simulations", 1)  # User can set >1
        num_workers = command_data.get("num_workers", max(1, mp.cpu_count() - 1))

        if num_simulations < 1:
            print("Number of simulations must be at least 1.")
            logging.error("Number of simulations must be at least 1.")
            return

        # Create a timestamped root folder inside corrupted_batch_path
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        corrupted_batch_root = os.path.join(corrupted_batch_path, timestamp)
        os.makedirs(corrupted_batch_root, exist_ok=True)
        print(f"Created timestamped corrupted root folder: {corrupted_batch_root}")
        logging.info(f"Created timestamped corrupted root folder: {corrupted_batch_root}")

        # Define the methods and approaches to traverse
        methods = ['nerfacto', 'splatfacto']
        approaches = ['rfdvc', 'vc']
        codecs = ['h264', 'h265']

        # Collect all tasks to be processed
        tasks = []

        # Traverse each method
        for method in methods:
            method_path = os.path.join(encoded_batch_path, method)
            if not os.path.isdir(method_path):
                print(f"Method directory not found: {method_path}")
                logging.warning(f"Method directory not found: {method_path}")
                continue
            print(f"Processing method: {method}")

            # Traverse each area within the method
            for area in os.listdir(method_path):
                area_path = os.path.join(method_path, area)
                if not os.path.isdir(area_path):
                    print(f"  Area directory not found: {area_path}")
                    logging.warning(f"Area directory not found: {area_path}")
                    continue
                print(f"  Processing area: {area}")

                # Traverse each traffic condition within the area
                for traffic in os.listdir(area_path):
                    traffic_path = os.path.join(area_path, traffic)
                    if not os.path.isdir(traffic_path):
                        print(f"    Traffic directory not found: {traffic_path}")
                        logging.warning(f"Traffic directory not found: {traffic_path}")
                        continue
                    print(f"    Processing traffic condition: {traffic}")

                    # Traverse each weather condition within the traffic condition
                    for weather in os.listdir(traffic_path):
                        weather_path = os.path.join(traffic_path, weather)
                        if not os.path.isdir(weather_path):
                            print(f"      Weather directory not found: {weather_path}")
                            logging.warning(f"Weather directory not found: {weather_path}")
                            continue
                        print(f"      Processing weather condition: {weather}")

                        # Traverse each approach within the weather condition
                        for approach in approaches:
                            approach_path = os.path.join(weather_path, approach)
                            if not os.path.isdir(approach_path):
                                print(f"        Approach directory not found: {approach_path}")
                                logging.warning(f"Approach directory not found: {approach_path}")
                                continue
                            print(f"        Processing approach: {approach}")

                            # Traverse each run-camera combination within the approach
                            for run_camera in os.listdir(approach_path):
                                run_camera_path = os.path.join(approach_path, run_camera)
                                if not os.path.isdir(run_camera_path):
                                    print(f"          Run-Camera directory not found: {run_camera_path}")
                                    logging.warning(f"Run-Camera directory not found: {run_camera_path}")
                                    continue

                                # Attempt to split run_camera into run and camera parts using the standalone function
                                run, camera = parse_run_camera(run_camera)
                                if not run or not camera:
                                    print(f"          Invalid Run-Camera name: {run_camera}")
                                    logging.warning(f"Invalid Run-Camera name: {run_camera}")
                                    continue
                                print(f"          Processing Run: {run}, Camera: {camera}")

                                # Traverse each batch within the run-camera directory
                                for batch in os.listdir(run_camera_path):
                                    batch_path = os.path.join(run_camera_path, batch)
                                    if not os.path.isdir(batch_path):
                                        print(f"            Batch directory not found: {batch_path}")
                                        logging.warning(f"Batch directory not found: {batch_path}")
                                        continue

                                    # Extract Batch_Size from batch directory name
                                    if not batch.startswith('batch_'):
                                        print(f"            Invalid batch name (missing 'batch_'): {batch}")
                                        logging.warning(f"Invalid batch name (missing 'batch_'): {batch}")
                                        continue
                                    try:
                                        batch_size = int(batch.split('_')[1])
                                    except (IndexError, ValueError):
                                        print(f"            Invalid batch size in batch name: {batch}")
                                        logging.warning(f"Invalid batch size in batch name: {batch}")
                                        continue
                                    print(f"            Processing batch: {batch} (Batch_Size: {batch_size})")

                                    # Traverse each resolution within the batch
                                    for resolution in os.listdir(batch_path):
                                        res_dir = os.path.join(batch_path, resolution)
                                        if not os.path.isdir(res_dir):
                                            print(f"              Resolution directory not found: {res_dir}")
                                            logging.warning(f"Resolution directory not found: {res_dir}")
                                            continue
                                        resolution_clean = resolution.replace('res_', '')
                                        print(f"              Processing resolution: {resolution_clean}")

                                        # Traverse each codec within the resolution
                                        for codec in codecs:
                                            codec_path = os.path.join(res_dir, codec)
                                            if not os.path.isdir(codec_path):
                                                print(f"                Codec directory not found: {codec_path}")
                                                logging.warning(f"Codec directory not found: {codec_path}")
                                                continue

                                            # Traverse each encoding parameter within the codec
                                            for encoding_param in os.listdir(codec_path):
                                                enc_param_dir = os.path.join(codec_path, encoding_param)
                                                if not os.path.isdir(enc_param_dir):
                                                    print(
                                                        f"                  Encoding Parameter directory not found: {enc_param_dir}")
                                                    logging.warning(
                                                        f"Encoding Parameter directory not found: {enc_param_dir}")
                                                    continue

                                                # Traverse encoded files within the encoding parameter directory
                                                for file in os.listdir(enc_param_dir):
                                                    file_path = os.path.join(enc_param_dir, file)
                                                    if os.path.isfile(file_path):
                                                        # Determine the codec from the file extension
                                                        file_ext = os.path.splitext(file)[1].lower()
                                                        if file_ext not in ['.h264', '.h265']:
                                                            # Skip non-encoded files
                                                            continue

                                                        print(
                                                            f"                    Queuing file for processing: {file}")

                                                        # Prepare task data for each simulation
                                                        for sim_num in range(1, num_simulations + 1):
                                                            task_data = {
                                                                'file_path': file_path,
                                                                'p_good_to_bad': p_good_to_bad,
                                                                'p_bad_to_good': p_bad_to_good,
                                                                'area': area,
                                                                'traffic': traffic,
                                                                'weather': weather,
                                                                'method': method,
                                                                'approach': approach,
                                                                'run': run,
                                                                'camera': camera,
                                                                'batch_size': batch_size,
                                                                'resolution': resolution_clean,
                                                                'codec': codec,
                                                                'encoding_param': encoding_param,
                                                                'file': file,
                                                                'simulation_number': sim_num,  # Simulation number
                                                                'num_simulations': num_simulations,
                                                                'corrupted_batch_path': corrupted_batch_root,
                                                                # Use timestamped root
                                                                'encoded_batch_path': encoded_batch_path
                                                                # To compute relative paths
                                                            }
                                                            tasks.append(task_data)

        if not tasks:
            print("No tasks to process.")
            return

        # Create a Manager for shared variables
        manager = mp.Manager()
        lock = manager.Lock()
        total_tasks = len(tasks)
        tasks_completed = manager.Value('i', 0)  # Shared integer value
        # total_time = manager.Value('d', 0.0)  # Shared float value for total time
        # queue = manager.Queue(maxsize=5)
        # queue = manager.Queue(maxsize=100)
        queue = manager.Queue()

        # Create the CSV file path with the same timestamp as corrupted_batch_root
        dataframe_filename = f"packet_loss_evaluation_{timestamp}.csv"
        dataframe_full_path = os.path.join(dataframes_path, dataframe_filename)

        # Create the CSV header based on data_entry keys
        header = [
            'Simulation_Number', 'Area', 'Traffic', 'Weather', 'RF_Model',
            'Approach', 'Run', 'Camera', 'Batch_Size', 'Resolution', 'Codec',
            'Encoding_Parameter', 'File_Name', 'Original_File_Size_Bytes',
            'Corrupted_File_Size_Bytes', 'Total_NAL_Units', 'Lost_NAL_Units',
            'Packet_Loss_Rate', 'BLER', 'SSIM'
        ]

        # Start the writer process
        writer_process = mp.Process(target=writer_function, args=(queue, dataframe_full_path, header))
        writer_process.start()

        # Use a multiprocessing Pool to process files in parallel
        pool = mp.Pool(processes=num_workers)
        # Map the tasks to the pool
        process_task_with_lock = partial(process_task, queue=queue, lock=lock, total_tasks=total_tasks,
                                         tasks_completed=tasks_completed)
        try:
            # Use map_async to handle exceptions properly
            result = pool.map_async(process_task_with_lock, tasks)

            # Wait for the result to finish and handle exceptions
            result.get()

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
            logging.info("Simulation interrupted by user.")
        except Exception as e:
            print(f"An error occurred during multiprocessing: {e}")
            logging.error(f"An error occurred during multiprocessing: {e}")
            traceback.print_exc()
        finally:
            pool.close()
            pool.join()
            # Signal the writer process to terminate
            queue.put('DONE')
            writer_process.join()
            print(f"\nData has been saved to CSV: {dataframe_full_path}")

# Example usage:
# Uncomment and modify the paths as needed to run the simulation.
# if __name__ == "__main__":
#     command_data = {
#         'encoded_batch_path': 'C:/Users/mDopiriak/Desktop/carla_city/encoded_files',    # Replace with your actual path
#         'dataframes_path': 'C:/Users/mDopiriak/Desktop/carla_city/dataframes',            # Replace with your desired CSV save path
#         'corrupted_batch_path': 'C:/Users/mDopiriak/Desktop/carla_city/corrupted_files',  # Replace with your desired corrupted files save path
#         'p_good_to_bad': 0.05,  # Probability of transitioning from Good to Bad state
#         'p_bad_to_good': 0.5,   # Probability of transitioning from Bad to Good state
#         'num_simulations': 10,   # Number of simulations per file
#         'num_workers': 4         # Number of parallel worker processes
#     }
#     manager = PacketLossManager()
#     manager.simulate_packet_loss(command_data)
