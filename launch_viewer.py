from model import gaussian
import argparse
import torch
from pathlib import Path
from typing import Optional
from viewer import Viewer, CameraState
import numpy as np
import time
from utils import load_camera_states


def load_gaussian_model(
    path: Path, iterations: Optional[int] = None
) -> gaussian.GaussianModel:
    cpt_lst = sorted([cpt for cpt in (path / "checkpoints").glob("*.pth")])
    if len(cpt_lst) == 0:
        raise ValueError("no checkpoint found")
    target_cpt = cpt_lst[-1]
    if iterations is not None:
        for cpt in cpt_lst:
            if cpt.name == f"iterations_{iterations}.pth":
                target_cpt = cpt
                break
        if target_cpt is None:
            raise ValueError(f"cannot find checkpoint for iteration {iterations}")

    gaussian_model: gaussian.GaussianModel = torch.load(target_cpt, map_location="cpu")
    gaussian_model = gaussian_model.eval().cuda()
    return gaussian_model


def waiting_exit():
    print("viewer is running, press Ctrl+C to exit")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.path)
    gaussian_model = load_gaussian_model(path)
    camera_states = load_camera_states(path)

    @torch.no_grad()
    def gs_render_func(camera_state: CameraState) -> np.ndarray:
        data = {
            "w2c": torch.tensor(camera_state.w2c, dtype=torch.float32, device="cuda"),
            "K": torch.tensor(camera_state.K, dtype=torch.float32, device="cuda"),
            "height": camera_state.height,
            "width": camera_state.width,
        }
        return gaussian_model(data)["render_img"].cpu().numpy()

    viewer = Viewer(gs_render_func, camera_states)
    waiting_exit()
