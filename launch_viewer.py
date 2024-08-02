import argparse
import torch
from pathlib import Path
from viewer import Viewer, CameraState
import numpy as np
import time
from utils import load_camera_states, load_gaussian_model


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
    parser.add_argument("--iterations", "-i", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    gaussian_model = load_gaussian_model(path, args.iterations).eval()
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
