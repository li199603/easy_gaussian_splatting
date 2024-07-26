import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore
import torch
from torch import Tensor
import torch.nn.functional as F


def k_nearest_neighbors_dists(points: np.ndarray, k: int) -> np.ndarray:
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(points)
    dists, _ = nn_model.kneighbors(points)
    return dists[:, 1:].astype(np.float32)


def to_sh_on_zero_degree(rgbs: np.ndarray) -> np.ndarray:
    C0 = 0.28209479177387814
    return (rgbs - 0.5) / C0


class LR_Scheduler:
    def __init__(self, lr_init: float, lr_final: float, max_steps: int) -> None:
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps

    def __call__(self, cur_step: int) -> float:
        t = min(1.0, cur_step / self.max_steps)
        lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return lr


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, but got {quat.shape[-1]}")
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, but got {quat.shape[-1]}")
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))
