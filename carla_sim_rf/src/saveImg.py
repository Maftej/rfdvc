import json
import math
import os
import queue
from datetime import datetime
from pathlib import Path
import cv2
from typing import Union

import carla
import numpy as np
from .actors import spectate_vehicle,def_fov,def_height,def_width
from .client import simulate_ticks
from .transform import TransformFile


class Img_manager:
    transform_file: TransformFile
    intristics_set: bool
    number_of_images: int

    def __init__(self,N_img) -> None:
        self.transform_file = TransformFile()
        self.transform_file.set_intrinsics(def_width, def_height, def_fov)
        self.intristics_set = True
        self.number_of_images = N_img
        

    def save_all_images(self,image_queues: list,run_number):
        """Save images all images"""
        print("Saving all images...")
        colmap_im_id = 1
        for i, iq in enumerate(image_queues):
            count = 1
            while not iq.empty():
                image = iq.get()
                self.transform_file.add_frame(image, f"run{run_number}cam{i+1}", count, colmap_im_id)
                count += 1
                colmap_im_id += 1

    def save_images(self,image_queues: list,numb_cameras:int,run_number: int):
        """Save images all images"""
        print("Saving n images on current path...")
        processed_images = [[] for _ in range(numb_cameras)]
        sync_lists = [list(q.queue) for q in image_queues]

        sync_list_length = len(sync_lists[0])
        product = round(sync_list_length / self.number_of_images)
        product_numbers = [i for i in range(0, sync_list_length, product)]

        for index, sync_list in enumerate(sync_lists):
            for product_number in product_numbers:
                processed_images[index].append(sync_list[product_number])

        colmap_im_id = 1
        for i, image_list in enumerate(processed_images):
            count = 1
            for image in image_list:
                self.transform_file.add_frame(image, f"run{run_number}cam{i + 1}", count, colmap_im_id)
                count += 1
                colmap_im_id += 1
        