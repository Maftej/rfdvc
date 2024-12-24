import json
import math
import os
from datetime import datetime
from pathlib import Path
import cv2
import carla
import numpy as np


class TransformFile:
    def __init__(self) -> None:
        self.frames = []
        self.intrinsics = {}
        self.output_dir = Path(os.curdir) / "runs" / datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.image_dir = self.output_dir / "images"
        self.intrinsics_set = False
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_dir.mkdir(exist_ok=True, parents=True)

    def export_transforms_json(self, file_path="transforms.json") -> Path:
        output_path = self.output_dir / file_path
        with open(output_path, "w+") as f:
            obj = {**self.intrinsics, "frames": self.frames}
            json.dump(obj, f, indent=2)
        return output_path

    def add_frame(self, image: carla.Image, camera_name: str, count: int, colmap_im_id: int):
        """Save and add frame to list of frames"""
        file_path = f"{self.image_dir}/{camera_name}_{count:03d}.jpg"
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        # save as RGB
        cv2.imwrite(file_path, image_data[:, :, :3])

        self.frames.append(
            {
                "file_path": f"./images/{file_path.split('/')[-1]}",
                "transform_matrix": self.carla_to_nerf(image.transform),
                "colmap_im_id": colmap_im_id,
            }
        )

    def carla_to_nerf(self, camera_transform: carla.Transform):
        """Convert carla.Transform to 4x4 matrix that can be used in NeRFs"""
        location = camera_transform.location
        rotation = camera_transform.rotation
        nerf_transform = carla.Transform(
            carla.Location(x=-location.z, y=location.x, z=location.y),
            carla.Rotation(pitch=rotation.yaw + 90, yaw=rotation.roll + 90, roll=-rotation.pitch),
        )
        return nerf_transform.get_matrix()

    def set_intrinsics(self, image_size_x: int, image_size_y: int, fov: float):
        """Compute and set COLMAP intrinsics"""
        focal_length = math.tan(math.radians(fov / 2))
        fl_x = (0.5 * image_size_x) / focal_length
        fl_y = fl_x

        self.intrinsics = {
            "w": image_size_x,
            "h": image_size_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": image_size_x / 2,
            "cy": image_size_y / 2,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
            "camera_model": "OPENCV",
        }
        self.intrinsics_set = True

#Transform 2 camera_path

def matrix_to_list(c2w, transform, scale):
    c2w = np.array(c2w)
    transform = np.array(transform)
    transformed_pose = np.matmul(transform, c2w)
    scaled_dim = transformed_pose[:, 3:] * scale
    combined = np.concatenate((transformed_pose[:, :3], scaled_dim), axis=1)
    out = [item for row in combined for item in row]
    out.append(0.0)
    out.append(0.0)
    out.append(0.0)
    out.append(1.0)
    return out

def transform_2_camera(dataParser,transform):
    dataparser = json.loads(open(dataParser).read())
    transforms = json.loads(open(transform).read())

    transform = dataparser["transform"]
    scale = dataparser["scale"]

    out = {
        "camera_type": "perspective",
        "render_height": 1080,
        "render_width": 1920,
        "fps": 5.0,
        "seconds": len(transforms),
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": [
            {
                "camera_to_world": matrix_to_list(c2w["transform_matrix"], transform, scale),
                "fov": 58.72,
                "aspect": 1,
            }
            for c2w in transforms["frames"]
        ],
    }

    outstr = json.dumps(out, indent=4)
    with open("camera_path.json", mode="w") as f:
        f.write(outstr)
