import argparse
from src.transform import transform_2_camera
import carla
import numpy as np
from src.client import *
from src.actors import *
from src.cameraPos import *
from src.saveImg import *
from src import nerfspawns
CAPTURE_NTH_TICK = 5

def record(TRAFFIC = 0 ,Pedestrians = 0 ,N_IMAGES = 0,Spectate:bool = True,Train:bool=True,Weather:"str|None" = None,spawn:int = 0):
        img_manager = Img_manager(N_IMAGES)
        spawns = nerfspawns.spawns[spawn]
        for run in range(len(spawns)):
            DISTANCE = spawns[run]["distance"]
            client = Client(sync=True, seed=24)
            if Weather:
                client.weather(Weather)
            camera_fps = 10 / CAPTURE_NTH_TICK
            ego = spawn_ego(x=spawns[run]["x"], y=spawns[run]["y"], yaw=spawns[run]["yaw"])
            if Train:
                camera_lists = camera_lists = ["Front","Front.y30","Front.y-30","Front.bottom","Front.mid.left-30.bottom","Front.mid.right30.bottom","Front.mid.right30.top","Front.mid.left30.top","Front.top"]
            else:
                camera_lists = ["Front","Front.y30","Front.y-30"]
            cameras = spawn_cameras(ego,camera_fps,camera_lists)
            simulate_ticks(15)
            time = None
            if(not (TRAFFIC == 0 or Pedestrians == 0)):
                client.VehicleNPC.extend(spawn_traffic(TRAFFIC,retries=200))
                client.VehicleNPC.extend(spawn_pedestrians(Pedestrians,retries=60))
            simulate_ticks(30)
            set_traffic_manager(ego=ego, speed=10, route=spawns[run]["route"])
            async_queues = [queue.Queue() for _ in cameras]
            sync_queues = [queue.Queue() for _ in cameras]
            for i, cam in enumerate(cameras):
                cam.listen(async_queues[i].put)

            # start recording
            time_count = 0
            image_count = 0
            distance_travelled = 0
            prev_location = ego.get_location()
            while ((time and time_count < time)or (DISTANCE and distance_travelled < DISTANCE)
            ):
                if Spectate:
                    spectate_vehicle(ego)

                simulate_ticks(CAPTURE_NTH_TICK)
                # wait for all cameras to produce an image
                for i, sync_queue in enumerate(sync_queues):
                    sync_queue.put(async_queues[i].get())

                image_count += 1
                time_count += CAPTURE_NTH_TICK / 10
                curr_location = ego.get_location()
                distance_travelled += get_distance_traveled(prev_location, curr_location)
                prev_location = curr_location

            # stop recording and save images
            for cam in cameras:
                cam.stop()
            print(
                f"Finished recording {image_count*len(cameras)} images, {time_count:.1f} seconds, {distance_travelled:.0f} meters"
            )
            
            if(N_IMAGES>1):
                img_manager.save_images(sync_queues,len(camera_lists),run)
            else:
                img_manager.save_all_images(sync_queues,run)
            client.reload()
        transform_file = img_manager.transform_file.export_transforms_json()
        print(f"Transforms saved to: {transform_file}")

def main():
    parser = argparse.ArgumentParser(description="Script created for Thesis project 2024")

    parser.add_argument("-t2c", "--transform2camerapath", action="store_true", default=False,
                        help="If set, script expects two additional arguments path to transform.json for new route  (-new_path) and model paths to dataparser_transforms.json (-existed_model)")
    parser.add_argument("-new_path", type=str,
                        help="Path to the new route transform.json (only required if -t2c is set)")
    parser.add_argument("-existed_model", type=str,
                        help="Path to the existing models dataparser_transforms.json (only required if -t2c is set)")
    parser.add_argument("-r","--record", action="store_true", default=False,help= "Carla sim needs to be running , ")

    # Define arguments with defaults
    parser.add_argument("-t", "--traffic", type=int, default=0, help="Number of vehicles in simulation")
    parser.add_argument("-p", "--pedestrians", type=int, default=0, help="Number of pedestrians in simulation")
    parser.add_argument(
        "-path", "--path", type=int, default=0, help="Path 0 - 21 , Test_paths = 19,20,21 , Red_Paths = 0-10 , Green = 11 - 18"
    )
    parser.add_argument(
        "-n",
        "--n_images",
        type=int,
        default=0,
        help="Number of images to capture during simulation",
    )
    parser.add_argument(
        "-s",
        "--spectate",
        type=bool,
        default=False,
        help="Start Carla with spectator mode",
    )
    parser.add_argument("-tr", "--train", action="store_true", default=False, help="Train mode captures 9 cameras")
    parser.add_argument(
        "-w", "--weather", type=str, default=None, choices=["ClearSunset", "CloudyNoon","MidRainyNoon", None], help="Weather condition"
    )
    args = parser.parse_args()

    if args.transform2camerapath:
        if not args.new_path or not args.existed_model:
            print("Error: Both -new_path and -existed_model arguments are required with -t2c")
            exit(1)
        # Process new_path and existed_model paths here
        print("-"*10)
        print()
        try:
            transform_2_camera(args.existed_model,args.new_path)
            print(f"New camera_path.json was created in {args.existed_model}")
        except:
            print("Error: Check the corect form of the input paths")
    elif args.record:
        print(f"Traffic: {args.traffic}")
        print(f"Pedestrians: {args.pedestrians}")
        print(f"Number of Images: {args.n_images}")
        print(f"Spectate Mode: {args.spectate}")
        print(f"Train Mode: {args.train}")
        print(f"Weather: {args.weather}")
        print(f"spawn: {args.path}")
        record(args.traffic,args.pedestrians,args.n_images,args.spectate,args.train,args.weather,args.path)



if __name__ == "__main__":
  main()
