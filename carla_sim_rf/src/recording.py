import json
import math
import os
import queue
from datetime import datetime
from pathlib import Path
import cv2
from typing import Optional, Union

import carla
import numpy as np

from .actors import *
from .client import *
from .transform import TransformFile
from .cameraPos import *

def create_camera_instances(camera_lists: list,camera_fps: float,type: "rgb|depth|BB|mask",ego: carla.Vehicle = None,):
    camera_list_instances = []
    for camera_type in camera_lists:
        camera = Camera_Types[camera_type]
        transofrm = carla.Transform(carla.Location(x=camera["x"], y=camera["y"], z=camera["z"]),
                                    carla.Rotation(yaw=camera["yaw"], pitch=camera["pitch"], roll=camera["roll"]))
        if type == "rgb":
            camera_list_instances.append(spawn_camera(transofrm,mount=ego, tick=1/camera_fps))
        elif type == "depth":
            camera_list_instances.append(spawn_depth_camera(transofrm,mount=ego, tick=1/camera_fps))
        elif type == "BB":
            camera_list_instances.append(spawn_camera(transofrm,mount=ego, tick=1/camera_fps))
        elif type == "mask":
            camera_list_instances.append(spawn_semantic_camera(transofrm,mount=ego, tick=1/camera_fps))
        
    return camera_list_instances



def record_ride_rgb(
    cam_list: list,
    nth_tick: int,
    n_images: "int|None" = None,
    time: "float|None" = None,
    distance: "float|None" = None,
    spectate: bool = False,
    ego: carla.Vehicle = None,
    sensor_types: list = None,
):
    """Record ride with given cameras"""
    # print recording info
    camera_fps = startMessage(nth_tick,n_images,time,distance,len(cam_list))
    # create cameras 
    if sensor_types == None:
        sensor_types = ["rgb","depth","BB","mask"]

    async_lists_rgb = None
    async_lists_depth = None
    async_lists_mask = None

    if "rgb" in sensor_types:
        camera_list_instances_rgb = create_camera_instances(cam_list,camera_fps,"rgb",ego)
        async_lists_rgb = [queue.Queue() for _ in cam_list]
        sync_depth_queues = [queue.Queue() for _ in cam_list]
        for i, cam in enumerate(camera_list_instances_rgb):
            cam.listen(async_lists_rgb[i].put)
    if "depth" in sensor_types:
        camera_list_instances_depth = create_camera_instances(cam_list,camera_fps,"depth",ego)
        cc = carla.ColorConverter.Depth
        async_lists_depth = [queue.Queue() for _ in cam_list]
        for i, cam in enumerate(camera_list_instances_depth):
            cam.listen(async_lists_depth[i].put)
    if "BB" in sensor_types:
        K = build_projection_matrix(1920,1080,90)
        camera_list_instances_BB = create_camera_instances(cam_list,camera_fps,"BB",ego)
        async_lists_BB = [queue.Queue() for _ in cam_list]
        for i, cam in enumerate(camera_list_instances_BB):
            cam.listen(async_lists_BB[i].put)
    if "mask" in sensor_types:
        camera_list_instances_mask = create_camera_instances(cam_list,camera_fps,"mask",ego)
        color_converter = carla.ColorConverter.CityScapesPalette
        async_lists_mask = [queue.Queue() for _ in cam_list]
        for i, cam in enumerate(camera_list_instances_mask):
            cam.listen(async_lists_mask[i].put)

    # start recording
    time_count = 0
    image_count = 0
    distance_travelled = 0
    prev_location = ego.get_location()

    transformfile = TransformFile()
    #default settings according to Actor -> camera
    transformfile.set_intrinsics(def_width,def_height,def_fov)
    colmap_id = 0

    while ((distance and distance_travelled < distance)):
        spectate_vehicle(ego)

        simulate_ticks(nth_tick)
        if async_lists_rgb:
            for i, queues in enumerate(async_lists_rgb):
                image = queues.get()
                save_image(image, f"cam{i+1}", image_count, colmap_id,transformfile.image_rgb_dir)
                transformfile.add_transform_matrix(image.transform,i,image_count,colmap_id)
                colmap_id+=1
        if async_lists_mask:
            for i, queues in enumerate(async_lists_mask):
                image = queues.get()
                mask_dir = transformfile.image_dir / "mask"
                mask_dir.mkdir(exist_ok=True, parents=True)
                path = f"{mask_dir}/cam{i+1}_{image_count}.png"
                image.save_to_disk(path,color_converter)
        if async_lists_depth:
            for i, queues in enumerate(async_lists_depth):
                image = queues.get()
                depth_dir = transformfile.image_dir / "depth"
                depth_dir.mkdir(exist_ok=True, parents=True)
                #save_image(image, f"cam{i+1}", image_count, colmap_id,depth_dir)
                path = f"{depth_dir}/cam{i+1}_{image_count}.png"
                image.save_to_disk(path,cc)
        if async_lists_BB:
            for i, queues in enumerate(async_lists_BB):
                ##image = queues.get()
                bb_dir = transformfile.image_dir / "BB"
                bb_dir.mkdir(exist_ok=True, parents=True)
                #save_image(image, f"cam{i+1}", image_count, colmap_id,bb_dir)
                file_path = f"{bb_dir}/cam{i+1}_{image_count:03d}.png"
                bounding = queues.get()
                cv2.imwrite(file_path,(drawBB(bounding,ego.get_world(),ego,camera_list_instances_BB[i],K)[:, :, :3]))    
        
        image_count += 1
        time_count += nth_tick / 10
        curr_location = ego.get_location()
        distance_travelled += get_distance_traveled(prev_location, curr_location)
        prev_location = curr_location

    # stop recording and save images
    for cam in camera_list_instances_rgb:
        cam.stop()
    #add_images(sync_depth_queues,transformfile)
    output_path = transformfile.export_transforms_json()
    print(f"Transforms saved to: {output_path}")

def record_ride(
    cam_list: list,
    nth_tick: int,
    n_images: "int|None" = None,
    time: "float|None" = None,
    distance: "float|None" = None,
    spectate: bool = False,
    ego: carla.Vehicle = None,
):
    """Record ride with given cameras"""
    # print recording info
    camera_fps = 10 / nth_tick
    if n_images or time:
        if n_images is None:
            n_images = int(time * camera_fps)  # type: ignore
        print(
            f"Recording {n_images} images per camera, {n_images / camera_fps:.1f} seconds at {camera_fps:.1f} FPS, {len(cam_list)} cameras"
        )
    else:
        print(f"Recording {distance} meters at {camera_fps:.1f} FPS, {len(cam_list)} cameras")

    # create queues for each camera
    async_queues = [queue.Queue() for _ in cam_list]
    sync_queues = [queue.Queue() for _ in cam_list]
    for i, cam in enumerate(cam_list):
        cam.listen(async_queues[i].put)

    
    # start recording
    time_count = 0
    image_count = 0
    distance_travelled = 0
    prev_location = ego.get_location()
    while (
        (time and time_count < time)
        or (distance and distance_travelled < distance)
    ):
        if spectate:
            spectate_vehicle(ego)

        simulate_ticks(nth_tick)
        # wait for all cameras to produce an image
        for i, sync_queue in enumerate(sync_queues):
            sync_queue.put(async_queues[i].get())

        image_count += 1
        time_count += nth_tick / 10
        curr_location = ego.get_location()
        distance_travelled += get_distance_traveled(prev_location, curr_location)
        prev_location = curr_location

    # stop recording and save images
    for cam in cam_list:
        cam.stop()
    if n_images != None:
        processed_images = [[] for _ in cam_list]
        sync_lists = [list(q.queue) for q in sync_queues]

        sync_list_length = len(sync_lists[0])
        product = round(sync_list_length / n_images)
        product_numbers = [i for i in range(0, sync_list_length, product)]

        for index, sync_list in enumerate(sync_lists):
            for product_number in product_numbers:
                processed_images[index].append(sync_list[product_number])
        
        save_images_lists(processed_images)
    else:
        save_images(sync_queues)

def save_images_lists(image_lists: list):
    """Save images from given queues"""
    print("Saving images...")
    transform_manager = TransformFile()
    intrinsics_set = False
    colmap_im_id = 1
    for i, image_list in enumerate(image_lists):
        count = 1
        for image in image_list:
        # image = iq.get()
            if not intrinsics_set:
                transform_manager.set_intrinsics(image.width, image.height, image.fov)
                intrinsics_set = True
            transform_manager.add_frame(image, f"camProtismer2{i + 1}", count, colmap_im_id)
            count += 1
            colmap_im_id += 1
        # print file path info
    output_path = transform_manager.export_transforms_json()
    print(f"Transforms saved to: {output_path}")
    
def save_images(image_queues: list):
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
            transform_file.add_frame(image, f"camProtismer2{i+1}", count, colmap_im_id)
            count += 1
            colmap_im_id += 1
    # print file path info
    output_path = transform_file.export_transforms_json()
    print(f"Transforms saved to: {output_path}")

def add_images(image_list: list,transform_file: TransformFile):
    """Save images from given queues"""
    print("Saving images...")
    colmap_im_id = 1
    for i, iq in enumerate(image_list):
        count = 1
        while not iq.empty():
            image = iq.get()
            transform_file.add_transform_matrix(image.transform,i, count, colmap_im_id)
            save_image(image, f"cam{i+1}", count, colmap_im_id,transform_file.image_dir)
            count += 1
            colmap_im_id += 1

def save_image(image: carla.Image, camera_name: str, count: int, colmap_im_id: int,image_dir: str):
    file_path = f"{image_dir}/{camera_name}_{count:03d}.png"
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_data = np.reshape(image_data, (image.height, image.width, 4))
    # save as RGB
    cv2.imwrite(file_path, image_data[:, :, :3])

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


#   Messages 
def startMessage(
    nth_tick: int,
    n_images: "int|None" = None,
    time: "float|None" = None,
    distance: "float|None" = None,
    cam_numbers:"int|None" = None,):
    # print recording info
    camera_fps = 10 / nth_tick
    if n_images or time:
        if n_images is None:
            n_images = int(time * camera_fps)  # type: ignore
        print(
            f"Recording {n_images} images per camera, {n_images / camera_fps:.1f} seconds at {camera_fps:.1f} FPS, {cam_numbers} cameras"
        )
    else:
        print(f"Recording {distance} meters at {camera_fps:.1f} FPS, {cam_numbers} cameras")
    return camera_fps