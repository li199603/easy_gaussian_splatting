import random
import numpy as np
import torch
import sys
from loguru import logger
from pathlib import Path
from viewer import CameraState
import json
from typing import Optional, List


def set_global_state(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    fmt = "<green>{time:MMDD-HH:mm:ss.SSSSSS}</green> | <level>{level:5}</level> | <level>{message}</level>"
    level = "DEBUG"
    log_config = {
        "handlers": [
            {"sink": sys.stdout, "format": fmt, "level": level, "enqueue": True}
        ]
    }
    logger.configure(**log_config)  # type: ignore


def load_camera_states(path: Path) -> List[CameraState]:
    camera_states = []
    with open(path / "cameras.json", "r") as f:
        for cam in json.load(f):
            c2w = np.eye(4)
            c2w[:3, :3] = np.array(cam["rotation"])
            c2w[:3, 3] = np.array(cam["position"])
            w2c = np.linalg.inv(c2w)
            K = np.array(
                [
                    [cam["fx"], 0, cam["width"] / 2],
                    [0, cam["fy"], cam["height"] / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            camera_states.append(CameraState(w2c, K, cam["width"], cam["height"]))
    return camera_states


def load_gaussian_model(
    path: Path, iterations: Optional[int] = None
) -> torch.nn.Module:
    cpt_lst = [cpt for cpt in (path / "checkpoints").glob("*.pth")]
    if iterations is not None:
        target_cpt = None
        for cpt in cpt_lst:
            if cpt.stem == f"iterations_{iterations}":
                target_cpt = cpt
                break
        if target_cpt is None:
            raise ValueError(f"cannot find checkpoint for iteration {iterations}")
    else:
        max_iterations = 0
        target_cpt = None
        for cpt in cpt_lst:
            iterations = int(cpt.stem.split("_")[1])
            if iterations > max_iterations:
                max_iterations = iterations
                target_cpt = cpt
        if target_cpt is None:
            raise ValueError("no checkpoint found")

    logger.info(f"load checkpoint from {target_cpt}")
    gaussian_model = torch.load(target_cpt, map_location="cpu").cuda()
    return gaussian_model
