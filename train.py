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
from scene import Scene


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
