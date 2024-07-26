import argparse
import yaml
import easydict  # type: ignore
from pathlib import Path
import sys
from loguru import logger
import random
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from scene import Scene
from model import GaussianModel


def train(cfg):
    project_name = Path(cfg.data).stem
    time_formatted = datetime.now().strftime(r"%m-%d_%H-%M-%S")
    cfg.output = str(Path(cfg.output) / project_name / time_formatted)
    Path(cfg.output).mkdir(parents=True)
    with open(Path(cfg.output) / "config.yaml", "w") as f:
        yaml.dump(dict(cfg), f, sort_keys=False)

    scene = Scene(
        cfg.data,
        cfg.data_format,
        cfg.output,
        cfg.num_iterations,
        cfg.eval,
        cfg.eval_split_ratio,
        cfg.use_masks,
    )
    train_dataloader = DataLoader(
        scene.train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.dataloader_workers,
        collate_fn=lambda x: x[0],
    )
    eval_dataloader = DataLoader(
        scene.eval_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=cfg.dataloader_workers,
        collate_fn=lambda x: x[0],
        persistent_workers=cfg.dataloader_workers != 0,
    )
    gaussian_model = GaussianModel(
        scene.pc,
        cfg.sh_degree,
        cfg.sh_degree_interval,
        cfg.means_lr_init,
        cfg.means_lr_final,
        cfg.means_lr_schedule_max_steps,
        cfg.densify_grad_thresh,
        cfg.densify_scale_thresh,
        cfg.num_splits,
        cfg.prune_radii_ratio_thresh,
        cfg.prune_scale_thresh,
        cfg.min_opacity,
        cfg.use_scale_regularization,
        cfg.max_scale_ratio,
    )


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


if __name__ == "__main__":
    set_global_state(seed=0, device="cuda:0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(f"config does not exist: {args.config}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"data does not exist: {args.data}")

    with open(args.config, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg["data"] = args.data
        cfg["output"] = args.output
    cfg = easydict.EasyDict(cfg)

    train(cfg)
