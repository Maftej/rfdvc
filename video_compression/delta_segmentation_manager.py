import os
import cv2
import glob
import numpy as np
# from skimage.rf_metrics import structural_similarity as compare_ssim
from skimage.segmentation import active_contour
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch
from functools import reduce
from ultralytics import YOLO, FastSAM, SAM

import torchvision.transforms.functional as tf
import torch
from utils.rf_metrics.lpips_pytorch import lpips
from utils.file_manager import FileManager

from skimage import exposure
import matplotlib.pyplot as plt

from skimage import metrics
from concurrent.futures import ThreadPoolExecutor


class DeltaSegmentationManager:
    def __init__(self):
        self.file_manager = FileManager()
        self.parallel_execution = False

    def get_mask(self, frame1, frame2, kernel=np.array((5, 5), dtype=np.uint8)):
        frame_diff = cv2.absdiff(frame1, frame2)
        th, mask = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def process_mask(self, frame1, frame2, mask_kernel=np.array((5, 5), dtype=np.uint8)):
        hist_eq_frame1 = cv2.equalizeHist(frame1)
        hist_eq_frame2 = cv2.equalizeHist(frame2)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        equalized_frame1 = clahe.apply(hist_eq_frame1)
        equalized_frame2 = clahe.apply(hist_eq_frame2)
        mask = self.get_mask(equalized_frame1, equalized_frame2, mask_kernel)
        return mask

    def draw_bboxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    def preprocess_images(self, imageA, imageB):
        imageA_corrected = self.gamma_correction(imageA, gamma=1.2)
        imageB_corrected = self.gamma_correction(imageB, gamma=1.2)
        imageA_filtered = self.bilateral_filtering(imageA_corrected)
        imageB_filtered = self.bilateral_filtering(imageB_corrected)
        return imageA_filtered, imageB_filtered

    def gamma_correction(self, image, gamma=1.0):
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_image = cv2.LUT(image, table)
        return corrected_image

    def bilateral_filtering(self, image, d=9, sigmaColor=75, sigmaSpace=75):
        filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        return filtered_image

    def optical_flow_mask(self, image_a, image_b):
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
        # hist_eq_frame1 = cv2.equalizeHist(image_a)
        # hist_eq_frame2 = cv2.equalizeHist(image_b)
        # clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        # equalized_frame1 = clahe.apply(hist_eq_frseame1)
        # equalized_frame2 = clahe.apply(hist_eq_frame2)
        # mask_kernel = np.array((5, 5))
        # mask1 = self.get_mask(equalized_frame1, equalized_frame2, mask_kernel)
        flow = cv2.calcOpticalFlowFarneback(image_a, image_b, None, 0.5, 20, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        threshold = 5
        _, diff_mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        # resulting_mask = self.merge_images(mask1, diff_mask)
        return diff_mask

    def remove_small_contours(self, img, min_contour_area=1500):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)
        return img

    def combine_images(self, bw_image, rgb_image):
        if bw_image.shape != rgb_image.shape[:2]:
            raise ValueError("Both images must have the same dimensions.")
        output_image = np.zeros_like(rgb_image)
        mask = bw_image == 255
        output_image[mask] = rgb_image[mask]
        return output_image

    def count_black_pixels_percentage(self, img):
        pixels = img.flatten()
        black_pixel_count = reduce(lambda acc, px: acc + 1 if px == 0 else acc, filter(lambda px: px == 0, pixels), 0)
        total_pixels = img.size
        black_pixel_percentage = (black_pixel_count / total_pixels) * 100
        return black_pixel_percentage

    def is_black(self, pixel):
        if len(pixel.shape) == 0:
            return pixel == 0
        elif len(pixel.shape) == 1:
            return np.array_equal(pixel[:3], [0, 0, 0])

    def merge_pixels(self, pixel1, pixel2):
        return pixel2 if self.is_black(pixel1) and not self.is_black(pixel2) else pixel1

    def merge_images(self, img1, img2):
        height, width = img1.shape[:2]
        combined_img = np.copy(img1)
        for y in range(height):
            for x in range(width):
                combined_img[y, x] = self.merge_pixels(img1[y, x], img2[y, x])
        return combined_img

    def remove_noise_masks(self, frames, max_distance=20, min_persistence=2):
        """
        Remove noise masks from a sequence of binary images by tracking blobs over multiple frames
        and checking if a mask is inside another mask.

        Parameters:
        - frames: List of binary images (numpy arrays) where white pixels represent masks.
        - max_distance: Maximum distance to consider two blobs the same across frames.
        - min_persistence: Minimum number of frames a blob must persist to be considered valid.

        Returns:
        - cleaned_frames: List of binary images with noise masks removed.
        """
        # Initialize trackers
        trackers = {}
        next_tracker_id = 1  # Start from 1 to avoid confusion with background label 0
        frame_height, frame_width = frames[0].shape
        num_frames = len(frames)

        # Store label images for reconstruction
        label_images = []

        for t in range(num_frames):
            frame = frames[t]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)

            blobs = []
            for label in range(1, num_labels):  # Skip background label 0
                blob = {
                    'label': label,
                    'area': stats[label, cv2.CC_STAT_AREA],
                    'centroid': centroids[label],
                    'bbox': stats[label, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP + 2],
                    'mask': (labels == label).astype(np.uint8)
                }
                blobs.append(blob)

            # Update trackers
            unmatched_trackers = set(trackers.keys())
            for blob in blobs:
                matched = False
                for tracker_id in unmatched_trackers:
                    tracker = trackers[tracker_id]
                    distance = np.linalg.norm(blob['centroid'] - tracker['centroid'][-1])
                    if distance < max_distance:
                        # Match found, update tracker
                        tracker['centroid'].append(blob['centroid'])
                        tracker['last_seen'] = t
                        tracker['age'] += 1
                        tracker['labels'].append(blob['label'])
                        matched = True
                        unmatched_trackers.remove(tracker_id)
                        break
                if not matched:
                    # New tracker
                    trackers[next_tracker_id] = {
                        'centroid': [blob['centroid']],
                        'first_seen': t,
                        'last_seen': t,
                        'age': 1,
                        'labels': [blob['label']]
                    }
                    next_tracker_id += 1

            # Remove stale trackers (not seen in the last frame)
            trackers_to_remove = [tracker_id for tracker_id, tracker in trackers.items() if
                                  t - tracker['last_seen'] > 0]
            for tracker_id in trackers_to_remove:
                del trackers[tracker_id]

            # Store label image for current frame
            label_images.append(labels)

        # Reconstruct cleaned frames
        cleaned_frames = []
        for t in range(num_frames):
            labels = label_images[t]
            frame_trackers = {}
            # Map labels to trackers for current frame
            for tracker_id, tracker in trackers.items():
                if tracker['first_seen'] <= t <= tracker['last_seen']:
                    label = tracker['labels'][t - tracker['first_seen']]
                    frame_trackers[label] = tracker

            # Create mask for persistent blobs
            persistent_mask = np.zeros_like(labels, dtype=np.uint8)
            for label, tracker in frame_trackers.items():
                if tracker['age'] >= min_persistence:
                    # Keep the blob
                    persistent_mask[labels == label] = 255
                else:
                    # Check if blob is inside another persistent blob
                    blob_mask = (labels == label).astype(np.uint8)
                    inside_persistent = False
                    for other_label in frame_trackers:
                        if other_label == label:
                            continue
                        other_tracker = frame_trackers[other_label]
                        if other_tracker['age'] >= min_persistence:
                            other_blob_mask = (labels == other_label).astype(np.uint8)
                            intersection = cv2.bitwise_and(blob_mask, other_blob_mask)
                            if np.array_equal(intersection, blob_mask):
                                inside_persistent = True
                                break
                    if inside_persistent:
                        persistent_mask[labels == label] = 255
                    # Else, blob is considered noise and not added to persistent_mask

            cleaned_frames.append(persistent_mask)

            # for cleaned_frame in cleaned_frames:
            #     cv2.imshow('Cleaned frame', cleaned_frame)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     exit(0)

        return cleaned_frames

    def process_binary_masks(self, gt_images_cv2, binary_masks):
        modified_binary_masks = binary_masks
        # modified_binary_masks = self.remove_noise_masks(frames=binary_masks)
        delta_frames = []
        for gt_image_cv2, modified_binary_mask in zip(gt_images_cv2, modified_binary_masks):
            delta_frames.append(cv2.bitwise_and(gt_image_cv2, gt_image_cv2, mask=modified_binary_mask))

        return delta_frames

    def segment_objects_dl(self, gt_images_cv2, rf_images_cv2):
        if self.parallel_execution:
            futures = []
            for index, gt_image_cv2 in enumerate(gt_images_cv2):
                futures.append(self.executor.submit(self.process_segmentation_dl, gt_image_cv2, rf_images_cv2[index]))
            binary_frames = [future.result() for future in futures]
        else:
            binary_frames = []
            for index, gt_image_cv2 in enumerate(gt_images_cv2):
                # if index == 15:
                #     break
                binary_frame = self.process_segmentation_dl(gt_image_cv2, rf_images_cv2[index])
                binary_frames.append(binary_frame)
        # exit(0)

        tracking_size = 10
        binary_frames_length = len(binary_frames)
        if tracking_size > binary_frames_length:
            tracking_size = binary_frames_length

        # if self.parallel_execution:
        #     futures = []
        #
        #     for i in range(0, binary_frames_length, tracking_size):
        #         futures.append(self.executor.submit(self.process_binary_masks, gt_images_cv2[i:i + tracking_size], binary_frames[i:i + tracking_size]))
        #     delta_frames = [future.result() for future in futures]
        # else:
        delta_frames = []
        for i in range(0, binary_frames_length, tracking_size):
            delta_frames.extend(self.process_binary_masks(gt_images_cv2[i:i + tracking_size], binary_frames[i:i + tracking_size]))
        # Save each frame in rec_frames as a .jpg file in the specified directory
        delta_output_directory = r"C:\Users\mDopiriak\Desktop\carla_city\seg_frames"
        os.makedirs(delta_output_directory, exist_ok=True)  # Create the directory if it doesn't exist
        print("DELTA FRAMES=", len(delta_frames))
        for i, frame in enumerate(delta_frames):
            output_path = os.path.join(delta_output_directory, f"delta_frame_{i}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        exit(0)
        return delta_frames

    def fast_show_mask_gpu(self, annotation):
        msak_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=False)
        annotation = annotation[sorted_indices]

        index = (annotation != 0).to(torch.long).argmax(dim=0)

        # https://github.com/taketwo/glasbey
        # GLASBEY = [(255,255,255), (0,0,255), (255,0,0), (0,255,0), (0,0,51), (255,0,182), (0,83,0), (255,211,0), (0,159,255), (154,77,66), (0,255,190), (120,63,193), (31,150,152), (255,172,253), (177,204,113), (241,8,92), (254,143,66), (221,0,255), (32,26,1), (114,0,85), (118,108,149), (2,173,36), (200,255,0), (136,108,0), (255,183,159), (133,133,103), (161,3,0), (20,249,255), (0,71,158), (220,94,147), (147,212,255), (0,76,255), (0,66,80), (57,167,106), (238,112,254), (0,0,100), (171,245,204), (161,146,255), (164,255,115), (255,206,113), (71,0,21), (212,173,197), (251,118,111), (171,188,0), (117,0,215), (166,0,154), (0,115,254), (165,93,174), (98,132,2), (0,121,168), (0,255,131), (86,53,0), (159,0,63), (66,45,66), (255,242,187), (0,93,67), (252,255,124), (159,191,186), (167,84,19), (74,39,108), (0,16,166), (145,78,109), (207,149,0), (195,187,255), (253,68,64), (66,78,32), (106,1,0), (181,131,84), (132,233,147), (96,217,0), (255,111,211), (102,75,63), (254,100,0), (228,3,127), (17,199,174), (210,129,139), (91,118,124), (32,59,106), (180,84,255), (226,8,210), (0,1,20), (93,132,68), (166,250,255), (97,123,201), (98,0,122), (126,190,58), (0,60,183), (255,253,0), (7,197,226), (180,167,57), (148,186,138), (204,187,160), (55,0,49), (0,40,1), (150,122,129), (39,136,38), (206,130,180), (150,164,196), (180,32,128), (110,86,180), (147,0,185), (199,48,61), (115,102,255), (15,187,253), (172,164,100), (182,117,250), (216,220,254), (87,141,113), (216,85,34), (0,196,103), (243,165,105), (216,255,182), (1,24,219), (52,66,54), (255,154,0), (87,95,1), (198,241,79), (255,95,133), (123,172,240), (120,100,49), (162,133,204), (105,255,220), (198,82,100), (121,26,64), (0,238,70), (231,207,69), (217,128,233), (255,211,209), (209,255,141), (36,0,3), (87,163,193), (211,231,201), (203,111,79), (62,24,0), (0,117,223), (112,176,88), (209,24,0), (0,30,107), (105,200,197), (255,203,255), (233,194,137), (191,129,46), (69,42,145), (171,76,194), (14,117,61), (0,30,25), (118,73,127), (255,169,200), (94,55,217), (238,230,138), (159,54,33), (80,0,148), (189,144,128), (0,109,126), (88,223,96), (71,80,103), (1,93,159), (99,48,60), (2,206,148), (139,83,37), (171,0,255), (141,42,135), (85,83,148), (150,255,0), (0,152,123), (255,138,203), (222,69,200), (107,109,230), (30,0,68), (173,76,138), (255,134,161), (0,35,60), (138,205,0), (111,202,157), (225,75,253), (255,176,77), (229,232,57), (114,16,255), (111,82,101), (134,137,48), (99,38,80), (105,38,32), (200,110,0), (209,164,255), (198,210,86), (79,103,77), (174,165,166), (170,45,101), (199,81,175), (255,89,172), (146,102,78), (102,134,184), (111,152,255), (92,255,159), (172,137,178), (210,34,98), (199,207,147), (255,185,30), (250,148,141), (49,34,78), (254,81,97), (254,141,100), (68,54,23), (201,162,84), (199,232,240), (68,152,0), (147,172,58), (22,75,28), (8,84,121), (116,45,0), (104,60,255), (64,41,38), (164,113,215), (207,0,155), (118,1,35), (83,0,88), (0,82,232), (43,92,87), (160,217,146), (176,26,229), (29,3,36), (122,58,159), (214,209,207), (160,100,105), (106,157,160), (153,219,113), (192,56,207), (125,255,89), (149,0,34), (213,162,223), (22,131,204), (166,249,69), (109,105,97), (86,188,78), (255,109,81), (255,3,248), (255,0,73), (202,0,35), (67,109,18), (234,170,173), (191,165,0), (38,44,51), (85,185,2), (121,182,158), (254,236,212), (139,165,89), (141,254,193), (0,60,43), (63,17,40), (255,221,246), (17,26,146), (154,66,84), (149,157,238), (126,130,72), (58,6,101), (189,117,101)]
        GLASBEY = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 51), (255, 0, 182), (0, 83, 0), (255, 211, 0),
                   (0, 159, 255), (154, 77, 66), (0, 255, 190), (120, 63, 193), (31, 150, 152), (255, 172, 253),
                   (177, 204, 113), (241, 8, 92), (254, 143, 66), (221, 0, 255), (32, 26, 1), (114, 0, 85),
                   (118, 108, 149), (2, 173, 36), (200, 255, 0), (136, 108, 0), (255, 183, 159), (133, 133, 103),
                   (161, 3, 0), (20, 249, 255), (0, 71, 158), (220, 94, 147), (147, 212, 255), (0, 76, 255),
                   (0, 66, 80), (57, 167, 106), (238, 112, 254), (0, 0, 100), (171, 245, 204), (161, 146, 255),
                   (164, 255, 115), (255, 206, 113), (71, 0, 21), (212, 173, 197), (251, 118, 111), (171, 188, 0),
                   (117, 0, 215), (166, 0, 154), (0, 115, 254), (165, 93, 174), (98, 132, 2), (0, 121, 168),
                   (0, 255, 131), (86, 53, 0), (159, 0, 63), (66, 45, 66), (255, 242, 187), (0, 93, 67),
                   (252, 255, 124), (159, 191, 186), (167, 84, 19), (74, 39, 108), (0, 16, 166), (145, 78, 109),
                   (207, 149, 0), (195, 187, 255), (253, 68, 64), (66, 78, 32), (106, 1, 0), (181, 131, 84),
                   (132, 233, 147), (96, 217, 0), (255, 111, 211), (102, 75, 63), (254, 100, 0), (228, 3, 127),
                   (17, 199, 174), (210, 129, 139), (91, 118, 124), (32, 59, 106), (180, 84, 255), (226, 8, 210),
                   (0, 1, 20), (93, 132, 68), (166, 250, 255), (97, 123, 201), (98, 0, 122), (126, 190, 58),
                   (0, 60, 183), (255, 253, 0), (7, 197, 226), (180, 167, 57), (148, 186, 138), (204, 187, 160),
                   (55, 0, 49), (0, 40, 1), (150, 122, 129), (39, 136, 38), (206, 130, 180), (150, 164, 196),
                   (180, 32, 128), (110, 86, 180), (147, 0, 185), (199, 48, 61), (115, 102, 255), (15, 187, 253),
                   (172, 164, 100), (182, 117, 250), (216, 220, 254), (87, 141, 113), (216, 85, 34), (0, 196, 103),
                   (243, 165, 105), (216, 255, 182), (1, 24, 219), (52, 66, 54), (255, 154, 0), (87, 95, 1),
                   (198, 241, 79), (255, 95, 133), (123, 172, 240), (120, 100, 49), (162, 133, 204), (105, 255, 220),
                   (198, 82, 100), (121, 26, 64), (0, 238, 70), (231, 207, 69), (217, 128, 233), (255, 211, 209),
                   (209, 255, 141), (36, 0, 3), (87, 163, 193), (211, 231, 201), (203, 111, 79), (62, 24, 0),
                   (0, 117, 223), (112, 176, 88), (209, 24, 0), (0, 30, 107), (105, 200, 197), (255, 203, 255),
                   (233, 194, 137), (191, 129, 46), (69, 42, 145), (171, 76, 194), (14, 117, 61), (0, 30, 25),
                   (118, 73, 127), (255, 169, 200), (94, 55, 217), (238, 230, 138), (159, 54, 33), (80, 0, 148),
                   (189, 144, 128), (0, 109, 126), (88, 223, 96), (71, 80, 103), (1, 93, 159), (99, 48, 60),
                   (2, 206, 148), (139, 83, 37), (171, 0, 255), (141, 42, 135), (85, 83, 148), (150, 255, 0),
                   (0, 152, 123), (255, 138, 203), (222, 69, 200), (107, 109, 230), (30, 0, 68), (173, 76, 138),
                   (255, 134, 161), (0, 35, 60), (138, 205, 0), (111, 202, 157), (225, 75, 253), (255, 176, 77),
                   (229, 232, 57), (114, 16, 255), (111, 82, 101), (134, 137, 48), (99, 38, 80), (105, 38, 32),
                   (200, 110, 0), (209, 164, 255), (198, 210, 86), (79, 103, 77), (174, 165, 166), (170, 45, 101),
                   (199, 81, 175), (255, 89, 172), (146, 102, 78), (102, 134, 184), (111, 152, 255), (92, 255, 159),
                   (172, 137, 178), (210, 34, 98), (199, 207, 147), (255, 185, 30), (250, 148, 141), (49, 34, 78),
                   (254, 81, 97), (254, 141, 100), (68, 54, 23), (201, 162, 84), (199, 232, 240), (68, 152, 0),
                   (147, 172, 58), (22, 75, 28), (8, 84, 121), (116, 45, 0), (104, 60, 255), (64, 41, 38),
                   (164, 113, 215), (207, 0, 155), (118, 1, 35), (83, 0, 88), (0, 82, 232), (43, 92, 87),
                   (160, 217, 146), (176, 26, 229), (29, 3, 36), (122, 58, 159), (214, 209, 207), (160, 100, 105),
                   (106, 157, 160), (153, 219, 113), (192, 56, 207), (125, 255, 89), (149, 0, 34), (213, 162, 223),
                   (22, 131, 204), (166, 249, 69), (109, 105, 97), (86, 188, 78), (255, 109, 81), (255, 3, 248),
                   (255, 0, 73), (202, 0, 35), (67, 109, 18), (234, 170, 173), (191, 165, 0), (38, 44, 51),
                   (85, 185, 2), (121, 182, 158), (254, 236, 212), (139, 165, 89), (141, 254, 193), (0, 60, 43),
                   (63, 17, 40), (255, 221, 246), (17, 26, 146), (154, 66, 84), (149, 157, 238), (126, 130, 72),
                   (58, 6, 101), (189, 117, 101)]
        GLASBEY = torch.tensor(GLASBEY) / 255.0
        # color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
        color = GLASBEY[:msak_sum].reshape(msak_sum, 1, 1, 3).to(annotation.device)

        transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 0.5
        visual = torch.cat([color, transparency], dim=-1)
        mask_image = torch.unsqueeze(annotation, -1) * visual

        show = torch.zeros((height, weight, 4)).to(annotation.device)
        h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

        show[h_indices, w_indices, :] = mask_image[indices]
        show_cpu = show.cpu().numpy()
        return show_cpu

    def compute_iou(self, masks1, masks2, iou_threshold=0.1):
        """
        Compute IoU between two sets of masks and return indices of overlapping masks.

        Args:
            masks1 (torch.Tensor): Tensor of shape (N1, H, W), binary masks on CPU.
            masks2 (torch.Tensor): Tensor of shape (N2, H, W), binary masks on CPU.
            iou_threshold (float): Threshold to consider masks as overlapping.

        Returns:
            masks1_to_remove (torch.Tensor): Indices of masks in masks1 to remove.
            masks2_to_remove (torch.Tensor): Indices of masks in masks2 to remove.
        """
        N1, H, W = masks1.shape
        N2 = masks2.shape[0]

        # Flatten masks for efficient computation
        masks1_flat = masks1.view(N1, -1).float()
        masks2_flat = masks2.view(N2, -1).float()

        # Compute intersection and union
        intersection = torch.mm(masks1_flat, masks2_flat.t())
        areas1 = masks1_flat.sum(dim=1).unsqueeze(1)
        areas2 = masks2_flat.sum(dim=1).unsqueeze(0)
        union = areas1 + areas2 - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        # Identify overlapping masks
        overlapping = (iou > iou_threshold).nonzero(as_tuple=False)

        if overlapping.numel() > 0:
            masks1_to_remove = overlapping[:, 0].unique()
            masks2_to_remove = overlapping[:, 1].unique()
        else:
            masks1_to_remove = torch.tensor([], dtype=torch.long)
            masks2_to_remove = torch.tensor([], dtype=torch.long)

        return masks1_to_remove, masks2_to_remove

    def remove_overlapping_masks(self, masks, masks_to_remove):
        """
        Remove masks based on indices.

        Args:
            masks (torch.Tensor): Tensor of masks.
            masks_to_remove (torch.Tensor): Indices to remove.

        Returns:
            torch.Tensor: Masks after removal.
        """
        if masks_to_remove.numel() > 0:
            mask = torch.ones(masks.shape[0], dtype=torch.bool)
            mask[masks_to_remove] = False
            return masks[mask]
        return masks

    def add_unique_masks_to_black_image(self, masks1, masks2, colors1, colors2, alpha=0.5):
        """
        Create a black image and add unique masks from masks1 and masks2 with specified colors and alpha.

        Args:
            masks1 (torch.Tensor): Tensor of shape (N1, H, W), binary masks on CPU.
            masks2 (torch.Tensor): Tensor of shape (N2, H, W), binary masks on CPU.
            colors1 (torch.Tensor): Tensor of shape (N1, 3), RGB colors normalized between 0 and 1 for masks1.
            colors2 (torch.Tensor): Tensor of shape (N2, 3), RGB colors normalized between 0 and 1 for masks2.
            alpha (float): Transparency factor.

        Returns:
            np.ndarray: RGBA image with unique masks overlaid on black background.
        """
        N1, H, W = masks1.shape
        N2 = masks2.shape[0]

        # Initialize a black RGBA image
        black_image = np.zeros((H, W, 4), dtype=np.float32)

        # Process masks1
        for i in range(N1):
            mask = masks1[i].numpy()  # Already on CPU
            mask = mask > 0  # Ensure binary

            color = colors1[i % len(colors1)].numpy()  # Get color, cycle if fewer colors
            color_rgba = np.array([*color, alpha], dtype=np.float32)  # RGBA

            # Create a colored mask image
            mask_image = np.zeros((H, W, 4), dtype=np.float32)
            mask_image[mask] = color_rgba

            # Alpha blending: overlay the mask on the black image
            black_image = mask_image + black_image * (1 - mask_image[:, :, 3:])

        # Process masks2
        for i in range(N2):
            mask = masks2[i].numpy()  # Already on CPU
            mask = mask > 0  # Ensure binary

            color = colors2[i % len(colors2)].numpy()  # Get color, cycle if fewer colors
            color_rgba = np.array([*color, alpha], dtype=np.float32)  # RGBA

            # Create a colored mask image
            mask_image = np.zeros((H, W, 4), dtype=np.float32)
            mask_image[mask] = color_rgba

            # Alpha blending: overlay the mask on the black image
            black_image = mask_image + black_image * (1 - mask_image[:, :, 3:])

        return black_image

    def segment_classes(self, image_input, classes_of_interest=[0, 1, 2, 3, 5, 7]):
        """
        Performs segmentation on the input image using the YOLO yolo_model and outputs a binary mask
        where the specified classes are white pixels and the rest are black.

        Parameters:
        - image_input: Input image as a NumPy array or file path (string).
        - model_path: Path to the YOLO yolo_model file.
        - classes_of_interest: List of class indices to include in the mask. If None, all classes are used.

        Returns:
        - binary_mask: The binary mask image as a NumPy array.
        """
        # Default classes if none are specified (COCO indices for the specified classes)
        if classes_of_interest is None:
            classes_of_interest = [0, 1, 2, 3, 5, 7]  # ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

        # Load the YOLO yolo_model
        yolo_model = YOLO('yolo11x-seg.pt')

        # Read the input image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
        else:
            img = image_input.copy()

        # Perform inference on the input image
        results = yolo_model(img, classes=classes_of_interest)

        # Get the original image dimensions
        img_height, img_width = img.shape[:2]

        # Create an empty black binary mask
        binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        # confidence_threshold = 0.7

        # Process the results
        for result in results:
            if result.masks is not None and hasattr(result.masks, 'data'):
                masks = result.masks.data  # Tensor of shape (N, h, w)
                # classes = result.boxes.cls  # Tensor containing class indices
                # confidences = result.boxes.conf

                for i in range(len(masks)):
                    # conf = confidences[i].item()  # Get the confidence score for this mask

                    # Filter out masks with low confidence scores
                    # if conf < confidence_threshold:
                    #     print("*"*40)
                    #     continue  # Skip this mask

                    # Convert mask to numpy array and move to CPU
                    mask = masks[i].cpu().numpy()  # Shape: (h, w)
                    # Resize mask to match the original image size
                    mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    mask_resized = (mask_resized > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)

                    # Multiply by 255 to get white pixels for the mask
                    mask_resized = mask_resized * 255

                    # Combine the masks using logical OR to handle overlapping masks
                    binary_mask = np.maximum(binary_mask, mask_resized)
            else:
                continue

        return binary_mask

    def process_segmentation_dl(self, gt_image_cv2, rf_image_cv2):
        """
        Process segmentation masks from two images, remove overlapping masks, and visualize unique masks.

        Args:
            gt_image_cv2 (np.ndarray): Ground truth image in BGR format.
            rf_image_cv2 (np.ndarray): Reference image in BGR format.

        Returns:
            np.ndarray: Binary image with unique masks overlaid on black background.
        """
        iou_threshold = 0.1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fast_sam_model = FastSAM("FastSAM-x.pt").to(device)

        # focus_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
        yolo_binary_mask = self.segment_classes(np.copy(gt_image_cv2))

        with torch.no_grad():
            # Process Ground Truth Image
            results = fast_sam_model(gt_image_cv2, device=device, retina_masks=True, imgsz=(640, 480), conf=0.4, iou=0.9)
            masks1 = results[0].masks.data  # Shape: (N1, H, W)
            print(f"Number of masks in GT image: {masks1.shape[0]}")

            # delta_output_directory = r"C:\Users\mDopiriak\Desktop\carla_city\seg_frames"
            # os.makedirs(delta_output_directory, exist_ok=True)  # Create the directory if it doesn't exist

            # Save each frame in rec_frames as a .jpg file in the specified directory
            # output_path = os.path.join(delta_output_directory, f"seg_frame1.jpg")
            # cv2.imwrite(output_path, gt_image_cv2)

            # Process Reference Image
            results2 = fast_sam_model(rf_image_cv2, device=device, retina_masks=True, imgsz=(640, 480), conf=0.4, iou=0.9)
            masks2 = results2[0].masks.data  # Shape: (N2, H, W)
            # print(f"Number of masks in RF image: {masks2.shape[0]}")
            # Save each frame in rec_frames as a .jpg file in the specified directory
            # output_path = os.path.join(delta_output_directory, f"seg_frame2.jpg")
            # cv2.imwrite(output_path, rf_image_cv2)
            # print("OUTPUT_PATH=", output_path)
            # print("MASK2=", masks2)
            # exit(0)
            #     print(f"Saved: {output_path}")

            # Binarize masks and convert to bool for memory efficiency
            masks1 = (masks1 > 0).bool()
            masks2 = (masks2 > 0).bool()

            # Move masks to CPU to save GPU memory
            masks1 = masks1.cpu()
            masks2 = masks2.cpu()

            # Compute IoU and get overlapping mask indices
            masks1_to_remove, masks2_to_remove = self.compute_iou(masks1, masks2, iou_threshold=iou_threshold)
            # print(f"Overlapping masks to remove from GT: {masks1_to_remove.numel()}")
            # print(f"Overlapping masks to remove from RF: {masks2_to_remove.numel()}")

            # Remove overlapping masks
            masks1 = self.remove_overlapping_masks(masks1, masks1_to_remove)
            masks2 = self.remove_overlapping_masks(masks2, masks2_to_remove)

            # print(f"Number of unique masks in GT image: {masks1.shape[0]}")
            # print(f"Number of unique masks in RF image: {masks2.shape[0]}")

            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Create a black image with the same size as the original input images
            # height, width = gt_image_cv2.shape[:2]
            # black_image = np.zeros((height, width), dtype=np.uint8)

            # Combine masks1 and masks2 (they are already binary) by adding them together
            combined_masks = masks1.sum(dim=0).cpu().numpy() + masks2.sum(dim=0).cpu().numpy()

            # Threshold to ensure the result is binary (0 or 1), where 1 represents the masks
            fast_sam_binary_mask = np.clip(combined_masks, 0, 1)

            # Multiply by 255 to create a white mask on a black background
            fast_sam_binary_mask = (fast_sam_binary_mask * 255).astype(np.uint8)

            # Ensure that masks are NumPy arrays of type uint8
            yolo_binary_mask = yolo_binary_mask.astype(np.uint8)
            fast_sam_binary_mask = fast_sam_binary_mask.astype(np.uint8)

            # Check that masks have the same dimensions
            if yolo_binary_mask.shape != fast_sam_binary_mask.shape:
                raise ValueError("Input masks must have the same dimensions")

            # Combine the masks: where yolo_binary_mask is 255, use it; else use binary_mask
            binary_mask = np.where(yolo_binary_mask == 255, 255, fast_sam_binary_mask)

            # Apply further processing, like removing small contours
            processed_mask = self.remove_small_contours(binary_mask, min_contour_area=1500)

            return processed_mask

    # def process_segmentation_dl(self, gt_image_cv2, rf_image_cv2):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print("DEVICE=", device)
    #     model = FastSAM("FastSAM-x.pt")
    #
    #     results = model(gt_image_cv2,
    #                    device=device,
    #                    retina_masks=True,
    #                    imgsz=(640, 480),
    #                    conf=0.4,
    #                    iou=0.9)
    #     #annotated_frame = results[0].plot(boxes=False)
    #     annotated_frame = self.fast_show_mask_gpu(results[0].masks.data)
    #     masks1 = results[0].masks.data
    #     print("NAMES=", results[0].names)
    #     # alpha = annotated_frame[:,:,3][:,:,None]
    #     # annotated_frame = (annotated_frame[:,:,:3]*255*alpha + gt_image_cv2*(1-alpha)) / 255
    #     # cv2.imshow("WINDOWS STRING", annotated_frame)
    #     # Wait for a key press indefinitely or for a set amount of time (in milliseconds)
    #     # cv2.waitKey(0)
    #     # Destroy the window after key press
    #     # cv2.destroyAllWindows()
    #
    #     results2 = model(rf_image_cv2,
    #                    device=device,
    #                    retina_masks=True,
    #                    imgsz=(640, 480),
    #                    conf=0.4,
    #                    iou=0.9)
    #     #annotated_frame = results[0].plot(boxes=False)
    #     annotated_frame2 = self.fast_show_mask_gpu(results2[0].masks.data)
    #     masks2 = results2[0].masks.data
    #     # alpha2 = annotated_frame2[:,:,3][:,:,None]
    #     # annotated_frame2 = (annotated_frame2[:,:,:3]*255*alpha2 + rf_image_cv2*(1-alpha2)) / 255
    #
    #     cv2.imshow("NEW FRAME", annotated_frame2)
    #     # cv2.imshow("WINDOWS STRING2", annotated_frame2)
    #     # Wait for a key press indefinitely or for a set amount of time (in milliseconds)
    #     cv2.waitKey(0)
    #     # Destroy the window after key press
    #     cv2.destroyAllWindows()
    #
    #
    #     # model = FastSAM("FastSAM-x.pt")
    #     #
    #     # results = model([gt_image_cv2])  # return a list of Results objects
    #    #
    #    # Process results list
    #     # for result in results:
    #     #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     #     masks = result.masks  # Masks object for segmentation masks outputs
    #     #     print("MASKS=", len(masks))
    #     #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     #     probs = result.probs  # Probs object for classification outputs
    #     #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     #     names = result.names
    #     #     print("names=", names)
    #         # result.show()  # display to screen
    #
    #     exit(0)
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    #     model.to(device)
    #     model.eval()
    #
    #     # Define the classes of interest
    #     target_classes = ['person', 'car', 'motorbike', 'bus', 'truck', 'bicycle']
    #
    #     results = model(gt_image_cv2)
    #
    #     # Get class names
    #     class_names = model.module.names if hasattr(model, 'module') else model.names
    #
    #     # Initialize mask with zeros (black image)
    #     mask = np.zeros(gt_image_cv2.shape[:2], dtype=np.uint8)
    #
    #     # Process detections
    #     for det in results.xyxy[0]:
    #         x1, y1, x2, y2, conf, cls_id = det
    #         class_name = class_names[int(cls_id)]
    #         if class_name in target_classes:
    #             # Convert coordinates to integers
    #             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #             # Draw filled rectangle on mask
    #             cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    #
    #     # Create a 3-channel mask
    #     mask_3ch = cv2.merge([mask, mask, mask])
    #
    #     # Apply the mask to the original image
    #     return cv2.bitwise_and(gt_image_cv2, mask_3ch)

    def segment_objects(self, gt_images_cv2, rf_images_cv2):
        if self.parallel_execution:
            futures = []
            for index, gt_image_cv2 in enumerate(gt_images_cv2):
                futures.append(self.executor.submit(self.process_segmentation, gt_image_cv2, rf_images_cv2[index]))
            detections = [future.result() for future in futures]
        else:
            detections = []
            for index, gt_image_cv2 in enumerate(gt_images_cv2):
                detections.append(self.process_segmentation(gt_image_cv2, rf_images_cv2[index]))
        return detections

    def segment_objects_gt_mask(self, gt_images_cv2, gt_mask_images_cv2):
        """
        Segments objects from a list of real images using corresponding mask images.

        Parameters:
        - gt_images_cv2: List of real images (as NumPy arrays in BGR format).
        - gt_mask_images_cv2: List of mask images (as NumPy arrays, white background and black mask).

        Returns:
        - segmented_images: List of images where the background is black, and the objects are from the original images.
        """
        segmented_images = []

        for real_image, mask_image in zip(gt_images_cv2, gt_mask_images_cv2):
            # Ensure the mask is in grayscale
            if len(mask_image.shape) == 3:
                mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_image

            # Ensure the mask is binary (0 and 255)
            _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

            # # Invert the mask so that the object areas are white
            # inverted_mask = cv2.bitwise_not(binary_mask)

            # Apply the inverted mask to the original image
            output_image = cv2.bitwise_and(real_image, real_image, mask=binary_mask)

            # Append the result to the list
            segmented_images.append(output_image)

        return segmented_images

    def process_segmentation(self, gt_image_cv2, rf_image_cv2):
        gt_image_cv2_copy, rf_image_cv2_copy = self.preprocess_images(gt_image_cv2, rf_image_cv2)
        diff_mask1 = self.optical_flow_mask(gt_image_cv2_copy, rf_image_cv2_copy)
        diff_mask2 = self.remove_small_contours(diff_mask1, min_contour_area=2000)
        final_image = self.combine_images(diff_mask2, gt_image_cv2)
        return final_image

    def set_parallel_execution(self, enabled):
        if enabled and not self.parallel_execution:
            self.executor = ThreadPoolExecutor()
        elif not enabled and self.parallel_execution:
            if hasattr(self, 'executor'):
                self.executor.shutdown()
                del self.executor
        self.parallel_execution = enabled
