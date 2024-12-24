import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DownStreamTaskManager:
    def __init__(self):
        pass

    def eval_downstream_task(self, command_data):
        print("EVAL DOWNSTREAM TASK")
        # Retrieve encoded batch path from command_data
        encoded_batch_path = command_data.get("encoded_batch_path", "")
        traffic_data = command_data.get("traffic", {})

        traffic_paths = self.extract_traffic_paths(traffic_data=traffic_data)
        encoded_file_paths = self.extract_encoded_batch_paths(encoded_batch_path=encoded_batch_path, traffic_data=traffic_data)

        self.traverse_combined_paths(traffic_paths, encoded_file_paths)

    def load_and_decode_videos(self, video_paths, fps=30):
        """
        Loads multiple .h264 or .h265 video files, decodes them, and extracts frames as images.

        Parameters:
            video_paths (list): List of paths to .h264 or .h265 video files.
            fps (int): Frames per second to read from each video. Default is 30 FPS.

        Returns:
            dict: A dictionary where keys are video paths and values are lists of frames extracted as images.
        """
        # video_frames = {}
        delta_frames = []

        for video_path in video_paths:
            # frames = []

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)

            if not video_capture.isOpened():
                print(f"Error: Unable to open video file {video_path}")
                continue  # Skip to the next video if this one cannot be opened

            # Set FPS for reading frames (for skipping frames if necessary)
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_skip = int(video_fps // fps) if video_fps > fps else 1

            frame_count = 0
            while True:
                # Read a frame
                ret, frame = video_capture.read()

                if not ret:
                    break  # Break if there are no more frames to read

                # Process every `frame_skip` frames to match the desired FPS
                if frame_count % frame_skip == 0:
                    delta_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            video_capture.release()

            # Store the list of frames in the dictionary with the video path as the key
            # delta_frames[video_path] = delta_frames

        return delta_frames

    def load_jpg_images_from_folder(self, folder_path):
        """
        Loads all .jpg images from a given folder using OpenCV.

        Parameters:
            folder_path (str): Path to the folder containing .jpg images.

        Returns:
            list: A list of images loaded as OpenCV objects.
        """
        images = []

        # Traverse files in the specified folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".jpg"):
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path)

                if image is not None:
                    images.append(image)
                else:
                    print(f"Warning: Could not load image {file_path}")

        return images

    def get_rec_frames(self, rf_frames, delta_frames):
        """
        Merges two lists of frames by overlaying non-black regions of delta_frames directly on top of rf_frames,
        avoiding any residual black pixels around the merged areas.

        Parameters:
            rf_frames (list): List of base frames (reference frames).
            delta_frames (list): List of frames with black regions representing areas to be added on top of rf_frames.

        Returns:
            list: A list of merged frames.
        """
        rec_frames = []

        # Check that both lists have the same length
        if len(rf_frames) != len(delta_frames):
            print("Error: rf_frames and delta_frames must have the same length.")
            return rec_frames

        for rf_frame, delta_frame in zip(rf_frames, delta_frames):
            # Create a mask where delta_frame has non-black pixels
            mask = cv2.inRange(delta_frame, (1, 1, 1), (255, 255, 255))

            # Convert the mask to 3 channels to match the frame dimensions
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Use the mask to copy non-black regions from delta_frame to rf_frame
            merged_frame = np.where(mask_3ch == 255, delta_frame, rf_frame)

            rec_frames.append(merged_frame)

        return rec_frames

    def traverse_combined_paths(self, traffic_paths, encoded_file_paths):
        """
        Traverses both traffic and encoded file paths in a single loop,
        printing organized information for each traffic type, weather, and codec.

        Parameters:
            traffic_paths (dict): A dictionary containing ground truth and radiance field paths for each traffic type and weather.
            encoded_file_paths (dict): A dictionary containing encoded file paths organized by traffic type, weather, and codec.
        """
        # Traverse both traffic and encoded file paths in a single loop
        for traffic_type, weathers in traffic_paths.items():
            for weather_name, paths in weathers.items():
                gt_path = paths.get("gt_dataset_path")
                rf_path = paths.get("rf_dataset_path")

                # Check if encoded file paths are available for the current traffic type and weather
                if traffic_type in encoded_file_paths and weather_name in encoded_file_paths[traffic_type]:
                    codecs_data = encoded_file_paths[traffic_type][weather_name]

                    # Specify the directory where the rec_frames will be saved
                    rec_output_directory = r"C:\Users\mDopiriak\Desktop\carla_city\rec_frames"
                    os.makedirs(rec_output_directory, exist_ok=True)  # Create the directory if it doesn't exist

                    delta_output_directory = r"C:\Users\mDopiriak\Desktop\carla_city\delta_frames"
                    os.makedirs(delta_output_directory, exist_ok=True)  # Create the directory if it doesn't exist

                    for codec, file_paths in codecs_data.items():
                        print(f"Traffic Type: {traffic_type}")
                        print(f"  Weather: {weather_name}")
                        print(f"    Ground Truth Path: {gt_path}")
                        print(f"    Radiance Field Path: {rf_path}")
                        print(f"    Codec: {codec}")
                        print(f"    Encoded Files:")

                        cav_frames = self.load_jpg_images_from_folder(gt_path)
                        rf_frames = self.load_jpg_images_from_folder(rf_path)

                        print("GT RF IMAGES LENGTH")
                        print("GT images length=", len(cav_frames))
                        print("RF images length=", len(rf_frames))
                        print("FILE PATHS LEN=", len(file_paths))
                        print("-" * 10)

                        print("DELTA FRAMES")
                        delta_frames = self.load_and_decode_videos(file_paths)
                        # Save each frame in rec_frames as a .jpg file in the specified directory
                        # for i, frame in enumerate(delta_frames):
                        #     output_path = os.path.join(delta_output_directory, f"delta_frame_{i}.jpg")
                        #     cv2.imwrite(output_path, frame)
                        #     print(f"Saved: {output_path}")
                        exit(0)
                        rec_frames = self.get_rec_frames(rf_frames, delta_frames)
                        self.evaluate_and_plot_comparison(cav_frames, rec_frames)

                        # # Save each frame in rec_frames as a .jpg file in the specified directory
                        # for i, frame in enumerate(rec_frames):
                        #     output_path = os.path.join(output_directory, f"rec_frame_{i}.jpg")
                        #     cv2.imwrite(output_path, frame)
                        #     print(f"Saved: {output_path}")

                        print("-" * 40)
                    return
                else:
                    print(f"No encoded file paths found for Traffic Type: {traffic_type}, Weather: {weather_name}")
                    print("-" * 40)
                return
            return

    def compute_iou(self, box1, box2):
        """Compute IoU for two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def compute_map(self, gt_boxes, pred_boxes, iou_threshold):
        """Compute mAP at a specific IoU threshold for bounding boxes."""
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)

        matched = [False] * len(gt_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_match = -1
            for i, gt_box in enumerate(gt_boxes):
                iou = self.compute_iou(pred_box, gt_box)
                if iou >= iou_threshold and iou > best_iou and not matched[i]:
                    best_iou = iou
                    best_match = i

            if best_match >= 0:
                true_positives += 1
                matched[best_match] = True
                false_negatives -= 1
            else:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return precision, recall

    def compute_mask_iou(self, mask1, mask2):
        """Compute IoU for two segmentation masks by accessing their binary data."""
        # Convert masks to binary arrays (1s and 0s) using their data attribute
        mask1_data = mask1.data > 0  # Convert to binary mask
        mask2_data = mask2.data > 0  # Convert to binary mask

        # Compute intersection and union
        intersection = torch.sum(mask1_data & mask2_data).item()
        union = torch.sum(mask1_data | mask2_data).item()

        return intersection / union if union > 0 else 0

    def compute_mask_map(self, gt_masks, pred_masks, iou_threshold):
        """Compute mAP at a specific IoU threshold for masks."""
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_masks)

        matched = [False] * len(gt_masks)

        for pred_mask in pred_masks:
            best_iou = 0
            best_match = -1
            for i, gt_mask in enumerate(gt_masks):
                iou = self.compute_mask_iou(pred_mask, gt_mask)
                if iou >= iou_threshold and iou > best_iou and not matched[i]:
                    best_iou = iou
                    best_match = i

            if best_match >= 0:
                true_positives += 1
                matched[best_match] = True
                false_negatives -= 1
            else:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return precision, recall

    def filter_boxes(self, result, focus_classes, min_area):
        """Filter detections by class and minimum size."""
        # return [det for det in detections if det.cls in focus_classes and (det.xywh[2] * det.xywh[3] >= min_size)]
        boxes = result.boxes
        print("BOXES LEN=", len(boxes))
        filtered_boxes = []
        for box in boxes:
            print(box.xyxy)
            print(box.xywh.cpu().numpy().flatten())
            print(box.conf)
            # print(result.names[box.cls])
            # print(box.cls)
            print(result.names[box.cls.item()])
            print("-"*30)
            box_xywh = box.xywh.cpu().numpy().flatten()
            width = box_xywh[2]
            height = box_xywh[3]
            area = width * height
            print("AREA=", area)
            if result.names[box.cls.item()] in focus_classes and (area >= min_area):
                filtered_boxes.append(box)
        print("FILTERED BOXES LEN=", len(filtered_boxes))
        return filtered_boxes

    def filter_masks(self, masks, focus_classes, min_size):
        """
        Filter segmentation masks by minimum area (pixel count) and specified focus classes.

        Parameters:
            masks (list): List of mask objects.
            min_size (int): Minimum area in pixels for a mask to be included.
            focus_classes (list): List of class names to filter masks by.

        Returns:
            list: Filtered list of masks.
        """
        filtered_masks = []
        for mask in masks:
            # Check if mask's class is in focus_classes and it meets the minimum size requirement
            if mask.cls in focus_classes and torch.sum(mask.data > 0).item() >= min_size:
                filtered_masks.append(mask)
        return filtered_masks

    def save_annotated_images(self, focus_classes, gt_img, rec_img, yolo_obj, yolo_seg, min_area=3500):
        """
        Saves annotated images with bounding boxes for object detection and segmentation masks for both
        ground truth (gt_img) and reconstructed (rec_img) images, applying a minimum size filter for each.

        Parameters:
            gt_img (numpy.ndarray): Ground truth image for annotation.
            rec_img (numpy.ndarray): Reconstructed image for annotation.
            yolo_obj (YOLO): YOLO model instance for object detection.
            yolo_seg (YOLO): YOLO model instance for segmentation.
            min_area (int): Minimum area in pixels for bounding boxes or masks to be drawn.
        """

        # Object Detection on Ground Truth Image
        gt_obj_det = yolo_obj(gt_img)
        filtered_gt_boxes = self.filter_boxes(gt_obj_det[0], focus_classes, min_area)
        gt_obj_det[0].boxes = filtered_gt_boxes  # Update with filtered boxes
        # Format ground truth
        gt_boxes = torch.tensor([det.xyxy.cpu().numpy() for det in gt_obj_det[0].boxes])
        gt_labels = torch.tensor([det.cls for det in gt_obj_det[0].boxes])

        ground_truth = {
            "boxes": gt_boxes,
            "labels": gt_labels
        }
        # print(gt_obj_det[0].boxes[0])
        # obj_detection_annotated_gt_img = gt_obj_det[0].plot()
        # cv2.imwrite("ground_truth_object_detection.jpg", obj_detection_annotated_gt_img)

        # Object Detection on Reconstructed Image
        rec_obj_det = yolo_obj(rec_img)
        filtered_rec_boxes = self.filter_boxes(rec_obj_det[0], focus_classes, min_area)
        rec_obj_det[0].boxes = filtered_rec_boxes  # Update with filtered boxes
        # obj_detection_annotated_rec_img = rec_obj_det[0].plot()
        # cv2.imwrite("reconstructed_object_detection.jpg", obj_detection_annotated_rec_img)

        # Format predictions
        rec_boxes = torch.tensor([det.xyxy.cpu().numpy() for det in rec_obj_det[0].boxes])
        rec_scores = torch.tensor([det.conf for det in rec_obj_det[0].boxes])
        rec_labels = torch.tensor([det.cls for det in rec_obj_det[0].boxes])

        prediction = {
            "boxes": rec_boxes,
            "scores": rec_scores,
            "labels": rec_labels
        }

        # # Segmentation on Ground Truth Image
        # gt_seg = yolo_seg(gt_img)
        # filtered_gt_masks = self.filter_masks(gt_seg[0].masks, focus_classes, min_size)
        # gt_seg[0].masks = filtered_gt_masks  # Update with filtered masks
        # segmentation_annotated_gt_img = gt_seg[0].plot()
        # cv2.imwrite("ground_truth_segmentation.jpg", segmentation_annotated_gt_img)
        #
        # # Segmentation on Reconstructed Image
        # rec_seg = yolo_seg(rec_img)
        # filtered_rec_masks = self.filter_masks(rec_seg[0].masks, focus_classes, min_size)
        # rec_seg[0].masks = filtered_rec_masks  # Update with filtered masks
        # segmentation_annotated_rec_img = rec_seg[0].plot()
        # cv2.imwrite("reconstructed_segmentation.jpg", segmentation_annotated_rec_img)

        print("Annotated images saved as:")
        print(" - 'ground_truth_object_detection.jpg'")
        print(" - 'reconstructed_object_detection.jpg'")
        # print(" - 'ground_truth_segmentation.jpg'")
        # print(" - 'reconstructed_segmentation.jpg'")

    def evaluate_and_plot_comparison(self, cav_frames, rec_frames, min_area=3500, our_iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
                                     focus_classes=["person", "car", "truck", "bus", "motorcycle", "bicycle"]):
        """
        Evaluates and compares object detection and segmentation between original cav_frames and reconstructed rec_frames
        using YOLOv11 models, and plots the results in separate plots.
        """
        # Initialize YOLO models for object detection and segmentation
        yolo_obj = YOLO('yolo11x.pt')  # Object detection model
        map_metric = MeanAveragePrecision(iou_thresholds=our_iou_thresholds)

        # yolo_seg = YOLO('yolo11x-seg.pt')  # Segmentation model

        # sample_gt_img = cav_frames[0]  # Ground truth frame
        # sample_rec_img = rec_frames[0]  # Reconstructed frame (optional)

        # Save annotated images
        # self.save_annotated_images(focus_classes, sample_gt_img, sample_rec_img, yolo_obj, yolo_seg)
        # return

        # Initialize results storage
        results = []

        # gt_boxes_all = []
        # rec_boxes_all = []
        for gt_img, rec_img in zip(cav_frames, rec_frames):
            # Run YOLO models on ground truth and reconstructed frames
            gt_obj_det = yolo_obj(gt_img)
            rec_obj_det = yolo_obj(rec_img)
            # gt_seg = yolo_seg(gt_img)
            # rec_seg = yolo_seg(rec_img)

            # Access detections and masks using the Results object attributes
            # gt_obj_det = gt_obj_det[0].boxes  # Bounding boxes for ground truth
            # rec_obj_det = rec_obj_det[0].boxes  # Bounding boxes for reconstructed frames
            # gt_seg = gt_seg[0].masks  # Segmentation masks for ground truth
            # rec_seg = rec_seg[0].masks  # Segmentation masks for reconstructed frames

            # Filter detections and masks by class and minimum size
            gt_obj_det_boxes = self.filter_boxes(gt_obj_det[0], focus_classes, min_area)

            # Ensure ground truth formatting
            if gt_obj_det_boxes:  # Only stack if there are detections
                gt_boxes = torch.stack([det.xyxy.cpu() for det in gt_obj_det_boxes]).reshape(-1, 4)
                gt_labels = torch.tensor([int(det.cls.item()) for det in gt_obj_det_boxes],
                                         dtype=torch.long)  # Ensure labels are integers
            else:
                gt_boxes = torch.empty((0, 4))  # Empty tensor with shape [0, 4] for boxes
                gt_labels = torch.empty((0,), dtype=torch.long)  # Empty tensor for labels

            ground_truth = {
                "boxes": gt_boxes,
                "labels": gt_labels
            }

            rec_obj_det_boxes = self.filter_boxes(rec_obj_det[0], focus_classes, min_area)

            # Ensure prediction data formatting with confidence scores
            if rec_obj_det_boxes:  # Only stack if there are detections
                rec_boxes = torch.stack([det.xyxy.cpu() for det in rec_obj_det_boxes]).reshape(-1, 4)
                rec_scores = torch.tensor([det.conf.item() for det in rec_obj_det_boxes]).reshape(-1)
                rec_labels = torch.tensor([int(det.cls.item()) for det in rec_obj_det_boxes],
                                          dtype=torch.long)  # Ensure labels are integers
            else:
                rec_boxes = torch.empty((0, 4))  # Empty tensor for boxes
                rec_scores = torch.empty((0,))  # Empty tensor for scores
                rec_labels = torch.empty((0,), dtype=torch.long)  # Empty tensor for labels

            prediction = {
                "boxes": rec_boxes,
                "scores": rec_scores,
                "labels": rec_labels
            }

            print("GROUND TRUTH=", ground_truth)
            print("-" * 40)
            print("PREDICTION=", prediction)

            map_metric.update(preds=[prediction], target=[ground_truth])
            # gt_seg = self.filter_masks(gt_seg, min_size)
            # rec_seg = self.filter_masks(rec_seg, min_size)

            # Convert detections to box format [x1, y1, x2, y2] for IoU computation
            # gt_boxes = [det.xyxy[0].cpu().numpy() for det in gt_obj_det]
            # rec_boxes = [det.xyxy[0].cpu().numpy() for det in rec_obj_det]

            # gt_boxes_all.extend(gt_boxes)
            # rec_boxes_all.extend(rec_boxes)
        results = map_metric.compute()
        print("mAP across all IoU thresholds:", results["map"])
        print(results)
        # Iterate over each specified IoU threshold and print the corresponding mAP
        # print("AP per class at each threshold:", results["map_per_class"])  # AP for each class if needed
        # print("AR (Average Recall):", results["mar"])  # AR if recall is also relevant
        # for class_name in focus_classes:
        #     for threshold in iou_thresholds:
        #         # Object detection mAP and IoU
        #         gt_boxes_class = [box for box in gt_boxes_all if box.cls == class_name]
        #         pred_boxes_class = [box for box in rec_boxes_all if box.cls == class_name]
        #
        #
        #         # Calculate AP for this class at the current IoU threshold
        #         # Segmentation mAP and IoU for pixel-wise masks
        #         # precision_seg, recall_seg = self.compute_mask_map(gt_seg, rec_seg, threshold)
        #         # gt_map_seg = precision_seg  # mAP for segmentation masks
        #         # gt_iou_seg = recall_seg  # IoU for segmentation masks
        #
        #         # Store all information in results
        #         results.append({
        #             "IoU_threshold": threshold,
        #             "mAP_object_detection": gt_map_obj,
        #             "IoU_object_detection": gt_iou_obj,
        #             # "mAP_segmentation": gt_map_seg,
        #             # "IoU_segmentation": gt_iou_seg
        #         })

        # Convert results to DataFrame and save to CSV
        # df_results = pd.DataFrame(results)
        # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # csv_path = f"eval_comparison_{timestamp}.csv"
        # df_results.to_csv(csv_path, index=False)
        # print(f"Results saved to {csv_path}")
        #
        # df_avg = df_results.groupby("IoU_threshold").mean(numeric_only=True).reset_index()
        #
        # # Plotting mAP on y-axis and IoU on x-axis
        # plt.figure(figsize=(12, 6))
        # plt.plot(df_avg["IoU_object_detection"], df_avg["mAP_object_detection"], marker='o', linestyle='-', color="b",
        #          label="Average mAP vs IoU")
        #
        # # Labels and title
        # plt.xlabel("Average IoU")
        # plt.ylabel("Average mAP")
        # plt.title("Object Detection: Average mAP vs Average IoU for All Classes")
        # plt.ylim(0, 1)
        # plt.xlim(df_avg["IoU_object_detection"].min(), df_avg["IoU_object_detection"].max())
        # plt.legend()
        # plt.grid(False)
        # plt.show()

        # Plot for Semantic Segmentation
        # plt.figure(figsize=(12, 6))
        # for threshold in iou_thresholds:
        #     subset = df_results[df_results["IoU_threshold"] == threshold]
        #
        #     plt.plot(subset["IoU_threshold"], subset["mAP_segmentation"], marker='^',
        #              label=f"Segmentation mAP at IoU={threshold}")
        #     plt.plot(subset["IoU_threshold"], subset["IoU_segmentation"], marker='d', linestyle='--',
        #              label=f"Segmentation IoU at IoU={threshold}")
        #
        # plt.xlabel("IoU Threshold")
        # plt.ylabel("Score")
        # plt.title("Semantic Segmentation: mAP and IoU for Pedestrians and Vehicles")
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    def extract_traffic_paths(self, traffic_data):
        """
        Extracts and organizes all paths from traffic_data.

        Parameters:
            traffic_data (dict): A dictionary containing traffic and weather information.

        Returns:
            dict: A dictionary organized by traffic type and weather, containing ground truth and radiance field paths.
        """
        # Initialize the storage structure for paths
        all_paths = {}

        # Traverse traffic types
        for traffic_type, conditions in traffic_data.items():
            if traffic_type not in all_paths:
                all_paths[traffic_type] = {}

            # Traverse weather conditions for each traffic type
            for condition in conditions:
                for weather in condition.get("weathers", []):
                    weather_name = weather.get("weather_name")
                    gt_path = weather.get("gt_dataset_path")
                    rf_path = weather.get("rf_dataset_path")

                    if weather_name and gt_path and rf_path:
                        # Create a sub-dictionary for each weather type
                        all_paths[traffic_type][weather_name] = {
                            "gt_dataset_path": gt_path,
                            "rf_dataset_path": rf_path
                        }

        return all_paths

    def extract_encoded_batch_paths(self, encoded_batch_path, traffic_data,
                               specified_weathers=("clear_noon", "clear_sunset", "wet_cloudy"),
                               methods=("nerfacto", "splatfacto"), approaches=("rfdvc",), codecs=("h264", "h265"),
                               resolution="res_1920x1080"):
        """
        Traverse folder structure and extract relevant paths for specified traffic and weather conditions.

        Parameters:
            encoded_batch_path (str): Root path where encoded batches are stored.
            traffic_data (dict): Dictionary containing traffic and weather information.
            specified_weathers (tuple): Weathers to consider (default is "clear_noon", "clear_sunset", "wet_cloudy").
            methods (tuple): Methods to include (default is "nerfacto" and "splatfacto").
            approaches (tuple): Approaches to consider (default is "rfdvc").
            codecs (tuple): Codecs to look for (default is "h264" and "h265").
            resolution (str): Resolution folder to filter by (default is "res_1920x1080").

        Returns:
            dict: A nested dictionary where keys are traffic types, weathers, and codecs, and values are sorted lists of file paths.
        """
        paths_info = {traffic_type: {weather: {codec: [] for codec in codecs} for weather in specified_weathers} for
                      traffic_type in traffic_data}

        # Start traversing through traffic data and weather conditions
        for traffic_type, traffic_conditions in traffic_data.items():
            for condition in traffic_conditions:
                for weather in condition.get("weathers", []):
                    weather_name = weather.get("weather_name")
                    if weather_name not in specified_weathers:
                        continue

                    gt_path = weather.get("gt_dataset_path")
                    rf_path = weather.get("rf_dataset_path")
                    if not gt_path or not rf_path:
                        print(f"Skipping missing path for traffic '{traffic_type}' and weather '{weather_name}'")
                        continue

                    for method in methods:
                        method_path = os.path.join(encoded_batch_path, method)
                        if not os.path.isdir(method_path):
                            print(f"Method directory not found: {method_path}")
                            continue

                        for area in os.listdir(method_path):
                            area_path = os.path.join(method_path, area)
                            if not os.path.isdir(area_path):
                                continue

                            traffic_path = os.path.join(area_path, traffic_type)
                            if not os.path.isdir(traffic_path):
                                continue

                            weather_path = os.path.join(traffic_path, weather_name)
                            if not os.path.isdir(weather_path):
                                continue

                            for approach in approaches:
                                approach_path = os.path.join(weather_path, approach)
                                if not os.path.isdir(approach_path):
                                    continue

                                for run_camera in os.listdir(approach_path):
                                    run_camera_path = os.path.join(approach_path, run_camera)
                                    if not os.path.isdir(run_camera_path):
                                        continue

                                    for batch in os.listdir(run_camera_path):
                                        batch_path = os.path.join(run_camera_path, batch)
                                        if not os.path.isdir(batch_path):
                                            continue

                                        # Check resolution folder
                                        res_dir = os.path.join(batch_path, resolution)
                                        if not os.path.isdir(res_dir):
                                            print(f"Resolution folder not found: {res_dir}")
                                            continue

                                        for codec in codecs:
                                            codec_path = os.path.join(res_dir, codec)
                                            if not os.path.isdir(codec_path):
                                                print(f"Codec directory not found: {codec_path}")
                                                continue

                                            for encoding_param in os.listdir(codec_path):
                                                enc_param_dir = os.path.join(codec_path, encoding_param)
                                                if not os.path.isdir(enc_param_dir):
                                                    continue

                                                # Gather and sort file paths
                                                file_paths = sorted(
                                                    [os.path.join(enc_param_dir, file) for file in
                                                     os.listdir(enc_param_dir) if
                                                     os.path.splitext(file)[1].lower() == f".{codec}"],
                                                    key=lambda x: os.path.basename(x)
                                                )

                                                # Append sorted files to paths_info dictionary
                                                paths_info[traffic_type][weather_name][codec].extend(file_paths)

        # Filter out empty lists to ensure only populated entries are returned
        paths_info = {
            traffic_type: {
                weather: {
                    codec: files for codec, files in codec_paths.items() if files
                } for weather, codec_paths in weather_paths.items() if any(codec_paths.values())
            } for traffic_type, weather_paths in paths_info.items() if
            any(any(codec_paths.values()) for codec_paths in weather_paths.values())
        }

        return paths_info
    # def eval_downstream_task(self, command_data):
    #     # Retrieve encoded batch path from command_data
    #     encoded_batch_path = command_data.get("encoded_batch_path", "")
    #     traffic_data = command_data.get("traffic", {})
    #
    #     # Initialize YOLO models for object detection and segmentation
    #     yolo_obj = YOLO('yolo11x.pt')
    #     yolo_seg = YOLO('yolo11x-seg.pt')
    #
    #     # Define IoU thresholds and classes to focus on
    #     iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    #     focus_classes = ['pedestrian', 'vehicle']
    #     results = []
    #
    #     # Extract relevant paths for each traffic type and weather
    #     paths_info = []
    #     specified_weathers = ["fog", "rain"]
    #
    #     for traffic_type, traffic_conditions in traffic_data.items():
    #         for condition in traffic_conditions:
    #             for weather in condition.get("weathers", []):
    #                 if weather.get("weather_name") in specified_weathers:
    #                     paths_info.append({
    #                         "traffic": traffic_type,
    #                         "weather_name": weather.get("weather_name"),
    #                         "gt_dataset_path": weather.get("gt_dataset_path"),
    #                         "rf_dataset_path": weather.get("rf_dataset_path")
    #                     })
    #
    #     # Define traversal parameters
    #     methods = ['nerfacto', 'splatfacto']
    #     approaches = ['rfdvc']  # Only RFDVC approach to be processed
    #     codecs = ['h264', 'h265']
    #
    #     # Traverse folder structure based on extracted paths and specified weathers
    #     for path_info in paths_info:
    #         traffic = path_info["traffic"]
    #         weather = path_info["weather_name"]
    #         gt_path = path_info["gt_dataset_path"]
    #         rf_path = path_info["rf_dataset_path"]
    #
    #         if not gt_path or not rf_path:
    #             print(f"Skipping missing path for traffic '{traffic}' and weather '{weather}'")
    #             continue
    #
    #         for method in methods:
    #             method_path = os.path.join(encoded_batch_path, method)
    #             if not os.path.isdir(method_path):
    #                 print(f"Method directory not found: {method_path}")
    #                 continue
    #
    #             for area in os.listdir(method_path):
    #                 area_path = os.path.join(method_path, area)
    #                 if not os.path.isdir(area_path):
    #                     continue
    #
    #                 traffic_path = os.path.join(area_path, traffic)
    #                 if not os.path.isdir(traffic_path):
    #                     continue
    #
    #                 weather_path = os.path.join(traffic_path, weather)
    #                 if not os.path.isdir(weather_path):
    #                     continue
    #
    #                 for approach in approaches:
    #                     approach_path = os.path.join(weather_path, approach)
    #                     if not os.path.isdir(approach_path):
    #                         continue
    #
    #                     for run_camera in os.listdir(approach_path):
    #                         run_camera_path = os.path.join(approach_path, run_camera)
    #                         if not os.path.isdir(run_camera_path):
    #                             continue
    #
    #                         for batch in os.listdir(run_camera_path):
    #                             batch_path = os.path.join(run_camera_path, batch)
    #                             if not os.path.isdir(batch_path):
    #                                 continue
    #
    #                             for resolution in os.listdir(batch_path):
    #                                 # Filter for 1920x1080 resolution only
    #                                 if resolution != 'res_1920x1080':
    #                                     continue
    #
    #                                 res_dir = os.path.join(batch_path, resolution)
    #                                 if not os.path.isdir(res_dir):
    #                                     continue
    #
    #                                 for codec in codecs:
    #                                     codec_path = os.path.join(res_dir, codec)
    #                                     if not os.path.isdir(codec_path):
    #                                         continue
    #
    #                                     for encoding_param in os.listdir(codec_path):
    #                                         enc_param_dir = os.path.join(codec_path, encoding_param)
    #                                         if not os.path.isdir(enc_param_dir):
    #                                             continue
    #
    #                                         # Process each .h264 or .h265 file
    #                                         for file in os.listdir(enc_param_dir):
    #                                             file_path = os.path.join(enc_param_dir, file)
    #                                             if os.path.splitext(file)[1].lower() not in ['.h264', '.h265']:
    #                                                 continue
    #
    #                                             # Decode video frames
    #                                             video = cv2.VideoCapture(file_path)
    #                                             frames = []
    #                                             success, frame = video.read()
    #                                             while success:
    #                                                 frames.append(frame)
    #                                                 success, frame = video.read()
    #                                             video.release()
    #
    #                                             # Process each frame for evaluation and record results
    #                                             for i, frame in enumerate(frames):
    #                                                 gt_img = cv2.imread(
    #                                                     os.path.join(gt_path, sorted(os.listdir(gt_path))[i]))
    #                                                 rf_background = cv2.imread(
    #                                                     os.path.join(rf_path, sorted(os.listdir(rf_path))[i]))
    #
    #                                                 # Mask out black pixels and merge with background
    #                                                 mask = frame != 0
    #                                                 merged = cv2.add(rf_background, frame,
    #                                                                  mask=mask.astype('uint8'))
    #
        #                                             # Run YOLO for object detection and segmentation
        #                                             gt_obj_det = yolo_obj(gt_img)
        #                                             rec_obj_det = yolo_obj(merged)
        #                                             gt_seg = yolo_seg(gt_img)
        #                                             rec_seg = yolo_seg(merged)
        #
        #                                             # Calculate mAP and IoU for each IoU threshold for the pedestrian and vehicle classes
        #                                             for threshold in iou_thresholds:
        #                                                 gt_map_obj = gt_obj_det.mAP(rec_obj_det, threshold,
        #                                                                             classes=focus_classes)
        #                                                 gt_iou_obj = gt_obj_det.IoU(rec_obj_det, threshold,
        #                                                                             classes=focus_classes)
        #                                                 gt_map_seg = gt_seg.mAP(rec_seg, threshold,
        #                                                                         classes=focus_classes)
        #                                                 gt_iou_seg = gt_seg.IoU(rec_seg, threshold,
        #                                                                         classes=focus_classes)
        #
        #                                                 # Store all information in results
        #                                                 results.append({
        #                                                     "method": method,
        #                                                     "area": area,
        #                                                     "traffic": traffic,
        #                                                     "weather": weather,
        #                                                     "approach": approach,
        #                                                     "run_camera": run_camera,
        #                                                     "batch": batch,
        #                                                     "resolution": resolution,
        #                                                     "codec": codec,
        #                                                     "encoding_param": encoding_param,
        #                                                     "IoU_threshold": threshold,
        #                                                     "mAP_object_detection": gt_map_obj,
        #                                                     "IoU_object_detection": gt_iou_obj,
        #                                                     "mAP_segmentation": gt_map_seg,
        #                                                     "IoU_segmentation": gt_iou_seg
        #                                                 })
        #
        # # Convert results to DataFrame and save to CSV
        # df_results = pd.DataFrame(results)
        # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # csv_path = f"eval_downstream_task_{timestamp}.csv"
        # df_results.to_csv(csv_path, index=False)
        # print(f"Results saved to {csv_path}")
        #
        # # Plot mAP and IoU for Pedestrians and Vehicles for specified weather
        # plt.figure(figsize=(12, 8))
        # for (traffic, codec), group in df_results.groupby(['traffic', 'codec']):
        #     plt.plot(group["IoU_threshold"], group["mAP_object_detection"], marker='o',
        #              label=f"{traffic} - {codec} (Object Detection mAP)")
        #     plt.plot(group["IoU_threshold"], group["IoU_object_detection"], marker='s', linestyle='--',
        #              label=f"{traffic} - {codec} (Object Detection IoU)")
        #     plt.plot(group["IoU_threshold"], group["mAP_segmentation"], marker='^',
        #              label=f"{traffic} - {codec} (Segmentation mAP)")
        #     plt.plot(group["IoU_threshold"], group["IoU_segmentation"], marker='d', linestyle='--',
        #              label=f"{traffic} - {codec} (Segmentation IoU)")
        #
        # plt.xlabel("IoU Threshold")
        # plt.ylabel("Score")
        # plt.title(f"mAP and IoU for Pedestrians and Vehicles - Weather: {weather}")
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.grid(True)
        # plt.show()
