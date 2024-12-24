import random
from typing import Literal
import carla
from .client import Client
from src.cameraPos import *

def_width,def_height,def_fov = 1920,1080,90

def set_traffic_manager(ego: carla.Actor,speed: int,route: list,):
    ego.set_autopilot(True) # Enable autopilot
    Client.traffic_manager.set_desired_speed(ego, speed)
    Client.traffic_manager.ignore_lights_percentage(ego, 100)
    Client.traffic_manager.ignore_signs_percentage(ego, 100)
    Client.traffic_manager.auto_lane_change(ego, False)
    Client.traffic_manager.set_route(ego, route * 6)


def spawn_camera(transform: carla.Transform,mount: carla.Actor = None,tick: float = 0.2) -> carla.Actor:
    """Spawn camera attached to given actor"""
    camera_bp = Client.blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(def_width))
    camera_bp.set_attribute("image_size_y", str(def_height))
    camera_bp.set_attribute("fov", str(def_fov))
    camera_bp.set_attribute("motion_blur_intensity", "0")
    camera_bp.set_attribute("sensor_tick", str(tick))  
    camera_transform = transform
    actor = Client.world.spawn_actor(camera_bp, camera_transform, attach_to=mount)
    return actor

def spawn_depth_camera(transform: carla.Transform,mount: carla.Actor = None,tick: float = 0.2) -> carla.Actor:
    """Spawn depth camera attached to given actor"""
    camera_bp = Client.blueprints.find("sensor.camera.depth")
    camera_bp.set_attribute("image_size_x", str(def_width))
    camera_bp.set_attribute("image_size_y", str(def_height))
    camera_bp.set_attribute("fov", str(def_fov))
    camera_bp.set_attribute("sensor_tick", str(tick))  
    camera_transform = transform
    actor = Client.world.spawn_actor(camera_bp, camera_transform, attach_to=mount)
    return actor

def spawn_semantic_camera(
    transform: carla.Transform,
    mount: carla.Actor = None,
    tick: float = 0.2,
) -> carla.Actor:
    """Spawn semantic_segmentation camera attached to given actor"""
    camera_bp = Client.blueprints.find("sensor.camera.semantic_segmentation")
    camera_bp.set_attribute("image_size_x", str(def_width))
    camera_bp.set_attribute("image_size_y", str(def_height))
    camera_bp.set_attribute("fov", str(def_fov))
    camera_bp.set_attribute("sensor_tick", str(tick))  
    camera_transform = transform
    actor = Client.world.spawn_actor(camera_bp, camera_transform, attach_to=mount)
    return actor

def spawn_ego(
    x: float = 0,
    y: float = 0,
    z: float = 0.5,
    yaw: float = 0,
) -> carla.Vehicle:
    """Spawn ego vehicle at given location"""
    vehicle_bp = Client.blueprints.find("vehicle.mini.cooper_s_2021")
    vehicle_transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw),
    )
    actor = Client.world.spawn_actor(vehicle_bp, vehicle_transform)
    return actor


def spawn_traffic(count: int, autopilot=True, retries=10) -> "list[carla.Vehicle]":
    """Spawn traffic vehicles at random locations"""
    actors: "list[carla.Vehicle]" = []
    spawn_points = Client.map.get_spawn_points()
    while len(actors) < count:
        spawn_point = random.choice(spawn_points)
        vehicle_bp = random.choice(Client.blueprints.filter("vehicle.*"))
        vehicle = Client.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            actors.append(vehicle)
            if autopilot:
                vehicle.set_autopilot(True)
        else:
            if retries == 0:
                break
            retries -= 1
    return actors


def spawn_pedestrians(count: int, retries=10) -> "list[carla.Walker]":
    """Spawn pedestrians at random locations"""
    actors: "list[carla.Walker]" = []
    spawn_points = Client.map.get_spawn_points()
    while len(actors) < count:
        spawn_point = random.choice(spawn_points)
        pedestrian_bp = random.choice(Client.blueprints.filter("walker.*"))
        pedestrian = Client.world.try_spawn_actor(pedestrian_bp, spawn_point)
        if pedestrian:
            if Client.sync:
                Client.world.tick()
            else:
                Client.world.wait_for_tick()
            controller_bp = Client.blueprints.find("controller.ai.walker")
            controller = Client.world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            controller.start()
            controller.go_to_location(Client.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())
            actors.append(controller)
            actors.append(pedestrian)
        else:
            if retries == 0:
                break
            retries -= 1
    return actors


def spectate_vehicle(vehicle: carla.Vehicle):
    """Set spectator position to given vehicle"""
    Client.spectator.set_transform(
        carla.Transform(
            carla.Location(
                x=vehicle.get_location().x,
                y=vehicle.get_location().y,
                z=vehicle.get_location().z + 20,
            ),
            carla.Rotation(
                pitch=-45,
                yaw=vehicle.get_transform().rotation.yaw,
            ),
        )
    )

def spawn_cameras(ego,camera_fps,camera_lists)-> "list[carla.Actor]":
    cam_list = []
    for camera_type in camera_lists:
        camera = Camera_Types[camera_type]
        transofrm = carla.Transform(carla.Location(x=camera["x"], y=camera["y"], z=camera["z"]),
                    carla.Rotation(yaw=camera["yaw"], pitch=camera["pitch"], roll=camera["roll"]))
        cam_list.append(spawn_camera(transofrm,mount=ego, tick=1/camera_fps))
    return cam_list

