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

from .actors import spectate_vehicle
from .client import *
from .transform import TransformFile

def record_ride_BoundBox(
    cam_list: list,
    nth_tick: int,
    n_images: "int|None" = None,
    time: "float|None" = None,
    distance: "float|None" = None,
    spectate: bool = False,
    ego: carla.Vehicle = None,
):
    camera_fps = 10 / nth_tick
  # create queues for each camera
    async_queues = [queue.Queue() for _ in cam_list]
    for i, cam in enumerate(cam_list):
        cam.listen(async_queues[i].put)

    # start recording
    time_count = 0
    image_count = 0
    distance_travelled = 0
    prev_location = ego.get_location()
    transformfile = TransformFile()
    #default settings according to Actor -> camera
    transformfile.set_intrinsics(1920,1080,90)
    #default settings according to Actor -> camera
    K = build_projection_matrix(1920,1080,90)
    colmap_id = 1
    while (
        (n_images and image_count < n_images)
        or (time and time_count < time)
        or (distance and distance_travelled < distance)
    ):
        if spectate:
            spectate_vehicle(ego)
        
        simulate_ticks(nth_tick)
        # wait for all cameras to produce an image
        for i, queues in enumerate(async_queues):
            bounding = queues.get()
            file_path = f"{transformfile.image_dir}/cam{i+1}_{image_count:03d}.png"
            cv2.imwrite(file_path,(drawBB(bounding,ego.get_world(),ego,cam_list[i],K)[:, :, :3]))
            transformfile.add_transform_matrix(bounding.transform,i,image_count,colmap_id)
            colmap_id+=1

        image_count += 1
        time_count += nth_tick / 10
        curr_location = ego.get_location()
        distance_travelled += get_distance_traveled(prev_location, curr_location)
        prev_location = curr_location

    # stop recording and save images
    for cam in cam_list:
        cam.stop()
    output_path = transformfile.export_transforms_json()
    print(f"Transforms saved to: {output_path}")
    

def drawBB(image: carla.Image,
           world: carla.World,
           vehicle:carla.Vehicle,
           camera: carla.Actor,
           K: np.array,):
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = np.reshape(image_data, (image.height, image.width, 4))
    
    for npc in world.get_actors().filter('*vehicle*'):
        # Filter out the ego vehicle
        if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)
            # Filter for the vehicles within 50m
            if dist < 50:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 1:
                    p1 = get_image_point(bb.location, K, world_2_camera) 
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = get_image_point(vert, K, world_2_camera)
                        if p[0] > x_max:
                            x_max = p[0]
                        if p[0] < x_min:
                            x_min = p[0]
                        if p[1] > y_max:
                            y_max = p[1]
                        if p[1] < y_min:
                            y_min = p[1]

                    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
    return img

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)
        # (x, y ,z) -> (y, -z, x)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]