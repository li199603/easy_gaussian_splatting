import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore


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
