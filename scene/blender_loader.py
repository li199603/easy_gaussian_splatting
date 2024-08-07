from .data_class import Frame, Pointcloud
from typing import Tuple, List
from pathlib import Path
import json
from PIL import Image
import numpy as np
from loguru import logger


def load_frames(
    path: Path,
    use_masks: bool,
    mask_expand_pixels: int,
    white_background: bool,
    suffix: str = ".png",
) -> List[Frame]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frames: List[Frame] = []
    with open(path, "r") as f:
        content = json.load(f)
        fov_x = content["camera_angle_x"]
        for frame_json in content["frames"]:
            file_name = frame_json["file_path"] + suffix  # type: str
            image_path = path.parent / file_name
            mask_dir = image_path.parent.parent / (image_path.parent.name + "_masks")
            mask_path = mask_dir / image_path.name
            image = Image.open(image_path)
            width, height = image.size
            fx = fy = width / (2 * np.tan(fov_x / 2))
            cx, cy = width / 2.0, height / 2.0
            # in nerf_synthetic, the camera's coordinate system is blender/opengl (X right, Y up, Z back)
            # need to convert to colmap/opencv (X right, Y down, Z forward)
            c2w = np.array(frame_json["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            frames.append(
                Frame(
                    image_path,
                    mask_path if mask_path.exists() and use_masks else None,
                    mask_expand_pixels,
                    width,
                    height,
                    fx,
                    fy,
                    cx,
                    cy,
                    w2c,
                    white_background,
                )
            )
    return frames


def generate_pointcloud(frames: List[Frame], num_points: int = 100000) -> Pointcloud:
    camera_positions = np.zeros((len(frames), 3))
    for i, frame in enumerate(frames):
        c2w = np.linalg.inv(frame.w2c)
        camera_positions[i] = c2w[:3, 3]
    max_val = camera_positions.max()
    min_val = camera_positions.min()
    center_val = (max_val + min_val) / 2.0
    min_val = center_val - (center_val - min_val) / 3
    max_val = center_val + (max_val - center_val) / 3

    xyzs = np.random.rand(num_points, 3) * (max_val - min_val) + min_val
    rgbs = np.ones((num_points, 3)) * 127.0
    rgbs = np.floor(rgbs).astype(np.uint8)
    pc = Pointcloud(xyzs, rgbs)
    return pc


def load_blender_data(
    path: str,
    use_masks: bool,
    mask_expand_pixels: int,
    eval: bool,
    eval_in_val: bool,
    eval_in_test: bool,
    white_background: bool,
) -> Tuple[List[Frame], Pointcloud, List[int], List[int]]:
    train_frames = load_frames(
        Path(path) / "transforms_train.json",
        use_masks,
        mask_expand_pixels,
        white_background,
    )
    eval_frames = []
    if eval_in_val:
        eval_frames += load_frames(
            Path(path) / "transforms_val.json",
            use_masks,
            mask_expand_pixels,
            white_background,
        )
    if eval_in_test:
        eval_frames += load_frames(
            Path(path) / "transforms_test.json",
            use_masks,
            mask_expand_pixels,
            white_background,
        )

    frames = eval_frames + train_frames
    num_frames = len(frames)
    split_point = len(eval_frames)
    indexes = list(range(num_frames))
    eval_indexes = indexes[:split_point]
    train_indexes = indexes[split_point:] if eval else indexes
    if len(eval_indexes) == 0:
        logger.warning("no data for evaluation")

    pc = generate_pointcloud(frames[split_point:] if eval else frames)
    return frames, pc, train_indexes, eval_indexes
