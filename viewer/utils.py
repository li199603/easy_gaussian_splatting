from viser.transforms import SE3
from typing import Tuple
import numpy as np


def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * np.arctan(pixels / (2 * focal))


def radians_norm(radians: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(radians), np.cos(radians))


class CameraState:
    def __init__(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> None:
        self.w2c = w2c  # colmap/opencv (X right, Y down, Z forward)
        self.K = K
        self.width = width
        self.height = height

    def fov(self) -> Tuple[float, float]:
        fov_x = focal2fov(self.K[0, 0], self.width)
        fov_y = focal2fov(self.K[1, 1], self.height)
        return fov_x, fov_y

    def roll_pitch_yaw(self) -> np.ndarray:
        rpy = SE3.from_matrix(self.w2c).inverse().rotation().as_rpy_radians()
        return np.array([rpy.roll, rpy.pitch, rpy.yaw], dtype=np.float32)

    def distance_to(self, other: "CameraState") -> float:
        self_c2w = np.linalg.inv(self.w2c)
        other_c2w = np.linalg.inv(other.w2c)
        dist = np.linalg.norm(self_c2w[:3, 3] - other_c2w[:3, 3], ord=2)
        return float(dist)
