from .data_class import Frame, Pointcloud
from typing import Tuple, Dict, Any, BinaryIO, List, Literal, Sequence
from pathlib import Path
from loguru import logger
import struct
import numpy as np
from pyquaternion import Quaternion  # type: ignore
import random


class Camera:
    def __init__(
        self,
        id: int,
        model_name: Literal["SIMPLE_PINHOLE", "PINHOLE"],
        width: int,
        height: int,
        params: Sequence[float],
    ):
        self.id = id
        self.model_name = model_name
        self.width = width
        self.height = height
        if model_name == "SIMPLE_PINHOLE":
            self.fx = params[0]
            self.fy = params[0]
            self.cx = params[1]
            self.cy = params[2]
        elif model_name == "PINHOLE":
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
        else:
            raise ValueError(f"unsupported camera model: {model_name}")

    def __str__(self) -> str:
        info = f"id: {self.id},"
        info += f" model_name: {self.model_name},"
        info += f" width: {self.width},"
        info += f" height: {self.height},"
        info += f" fx: {self.fx},"
        info += f" fy: {self.fy},"
        info += f" cx: {self.cx},"
        info += f" cy: {self.cy}"
        return info


class Image:
    def __init__(
        self,
        id: int,
        image_file_name: str,
        camera_id: int,
        quat: Sequence[float],
        trans: Sequence[float],
    ):
        self.id = id
        self.image_file_name = image_file_name
        self.camera_id = camera_id
        # quat and trans are for w2c
        self.quat = quat  # wxyz
        self.trans = trans

    def __str__(self) -> str:
        info = f"id: {self.id},"
        info += f" image_file_name: {self.image_file_name},"
        info += f" camera_id: {self.camera_id},"
        info += f" quat: {self.quat},"
        info += f" trans: {self.trans}"
        return info


def read_next_bytes(
    f: BinaryIO, num_bytes: int, format_char_sequence: str, endian_character="<"
) -> Tuple[Any, ...]:
    data = f.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def load_intrinsics_binary(path: Path) -> Dict[int, Camera]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    # {cam_model_id: [cam_model_name, num_params]}
    CAM_MAP = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
    }
    camera_map: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            properties = read_next_bytes(f, 24, "iiQQ")
            camera_id = properties[0]
            model_id = properties[1]
            width = properties[2]
            height = properties[3]
            if model_id not in CAM_MAP:
                raise ValueError(f"unsupported camera model id: {model_id}")
            model_name = CAM_MAP[model_id][0]
            num_params = CAM_MAP[model_id][1]
            params = read_next_bytes(f, 8 * num_params, "d" * num_params)
            camera_map[camera_id] = Camera(camera_id, model_name, width, height, params)  # type: ignore
    assert len(set([cam.model_name for cam in camera_map.values()])) == 1
    return camera_map


def load_extrinsics_binary(path: Path) -> Dict[int, Image]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    image_map: Dict[int, Image] = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            properties = read_next_bytes(f, 64, "idddddddi")
            image_id = properties[0]
            quat = properties[1:5]  # wxyz
            trans = properties[5:8]
            camera_id = properties[8]
            image_file_name = ""
            current_char = read_next_bytes(f, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_file_name += current_char.decode("utf-8")
                current_char = read_next_bytes(f, 1, "c")[0]
            # This part of the data involves the matching of points on the images and the 3D space.
            # It is useless here.
            _ = read_next_bytes(f, 8, "Q")[0]
            _ = read_next_bytes(f, 24 * _, "ddq" * _)
            image_map[image_id] = Image(
                image_id, image_file_name, camera_id, quat, trans
            )
    return image_map


def load_pointcloud(path: Path) -> Pointcloud:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "rb") as f:
        num_points = read_next_bytes(f, 8, "Q")[0]
        xyzs = np.empty((num_points, 3), dtype=np.float32)
        rgbs = np.empty((num_points, 3), dtype=np.uint8)
        for i in range(num_points):
            properties = read_next_bytes(f, 43, "QdddBBBd")
            xyz = np.array(properties[1:4])
            rgb = np.array(properties[4:7])
            # This part of the data involves the matching of points on the images and the 3D space.
            # It is useless here.
            _ = read_next_bytes(f, 8, "Q")[0]
            _ = read_next_bytes(f, 8 * _, "ii" * _)
            xyzs[i] = xyz
            rgbs[i] = rgb
        return Pointcloud(xyzs, rgbs)


def load_colmap_data(
    path: str,
    use_masks: bool,
    mask_expand_pixels: int,
    eval: bool,
    eval_split_ratio: float,
    white_background: bool,
) -> Tuple[List[Frame], Pointcloud, List[int], List[int]]:
    intrinsics_path = Path(path) / "sparse" / "0" / "cameras.bin"
    extrinsics_path = Path(path) / "sparse" / "0" / "images.bin"
    pointcloud_path = Path(path) / "sparse" / "0" / "points3D.bin"
    camera_map = load_intrinsics_binary(intrinsics_path)
    image_map = load_extrinsics_binary(extrinsics_path)
    pc = load_pointcloud(pointcloud_path)

    frames: List[Frame] = []
    mask_count = 0
    for image_id in image_map:
        image = image_map[image_id]
        camera = camera_map[image.camera_id]
        w2c = np.eye(4, dtype=np.float32)  # colmap/opencv (X right, Y down, Z forward)
        w2c[:3, :3] = Quaternion(image.quat).rotation_matrix
        w2c[:3, 3] = np.array(image.trans, dtype=np.float32)
        image_path = Path(path) / "images" / image.image_file_name
        mask_path = (Path(path) / "masks" / image.image_file_name).with_suffix(".png")
        frames.append(
            Frame(
                image_path,
                mask_path if mask_path.exists() and use_masks else None,
                mask_expand_pixels,
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                w2c,
                white_background,
            )
        )
        if frames[-1].mask_path is not None:
            mask_count += 1
    frames.sort(key=lambda frame: frame.image_path)
    msg = f"colmap data: {len(camera_map)} cameras, {len(image_map)} images, {pc.nbr_points} points"
    if use_masks:
        msg += f", {mask_count} masks"
    logger.info(msg)

    num_frames = len(frames)
    indexes = list(range(num_frames))
    random.shuffle(indexes)
    split_point = int(num_frames * eval_split_ratio)
    eval_indexes = indexes[:split_point]
    train_indexes = indexes[split_point:] if eval else indexes
    if len(eval_indexes) == 0:
        logger.warning("no data for evaluation")
    return frames, pc, train_indexes, eval_indexes
