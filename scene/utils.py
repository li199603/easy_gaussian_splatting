import numpy as np


def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * np.arctan(pixels / (2 * focal))
