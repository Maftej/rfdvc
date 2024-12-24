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
from .actors import *
from .client import *
from .transform import TransformFile

def record_ride_depth(
    cam_list: list,
    cam_depth_list: list,
    nth_tick: int,
    n_images: "int|None" = None,
    time: "float|None" = None,
    distance: "float|None" = None,
    spectate: bool = False,
    ego: carla.Vehicle = None,
):
    camera_fps = 10 / nth_tick
    cc = carla.ColorConverter.Depth

    # create queues for each camera
    async_queues = [queue.Queue() for _ in cam_list]
    sync_queues = [queue.Queue() for _ in cam_list]
    for i, cam in enumerate(cam_list):
        cam.listen(async_queues[i].put)

    async_depth_queues = [queue.Queue() for _ in cam_depth_list]
    sync_depth_queues = [queue.Queue() for _ in cam_depth_list]
    for i, cam in enumerate(cam_depth_list):
        cam.listen(async_depth_queues[i].put,carla.Depth)

    # start recording
    time_count = 0
    image_count = 0
    distance_travelled = 0
    prev_location = ego.get_location()
    while (
        (n_images and image_count < n_images)
        or (time and time_count < time)
        or (distance and distance_travelled < distance)
    ):
        if spectate:
            spectate_vehicle(ego)

        simulate_ticks(nth_tick)
        # wait for all cameras to produce an image
        for i, sync_queue in enumerate(sync_queues):
            sync_queue.put(async_queues[i].get())

        
        for i, sync_depth_queue in enumerate(sync_depth_queues):
            sync_depth_queue.put(async_depth_queues[i].get())

        image_count += 1
        time_count += nth_tick / 10
        curr_location = ego.get_location()
        distance_travelled += get_distance_traveled(prev_location, curr_location)
        prev_location = curr_location

    # stop recording and save images
    for cam in cam_list:
        cam.stop()
    save_images(sync_queues)
    save_images(sync_depth_queues,True)

def get_distance_traveled(previous_location: carla.Location, current_location: carla.Location) -> float:
    return np.sqrt(
        (current_location.x - previous_location.x) ** 2
        + (current_location.y - previous_location.y) ** 2
        + (current_location.z - previous_location.z) ** 2
    )  # type: ignore


def save_images(image_queues: list,depth:bool):
    """Save images from given queues"""
    print("Saving images...")
    transform_file = TransformFile()
    intrinsics_set = False
    colmap_im_id = 1
    for i, iq in enumerate(image_queues):
        count = 1
        while not iq.empty():
            image = iq.get()
            if not intrinsics_set:
                transform_file.set_intrinsics(image.width, image.height, image.fov)
                intrinsics_set = True
            transform_file.add_frame(image, f"cam{i+1}", count, colmap_im_id)
            count += 1
            colmap_im_id += 1
    # print file path info
    output_path = transform_file.export_transforms_json()
    print(f"Transforms saved to: {output_path}")

