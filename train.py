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
from scene import Scene, data_to_device
from model import gaussian
import tqdm
from torch.utils.tensorboard import SummaryWriter
from eval import Evaluator
from typing import Dict, Any
import time
from eval import eval


def train(cfg: easydict.EasyDict):
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
    gaussian_model = gaussian.GaussianModel(
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
    optimizer = gaussian.build_optimizers(
        gaussian_model,
        cfg.means_lr_init,
        cfg.log_scales_lr,
        cfg.quats_lr,
        cfg.sh_0_lr,
        cfg.sh_rest_lr,
        cfg.logit_opacities_lr,
    )
    loss_computer = gaussian.LossComputer(
        gaussian_model, cfg.lambda_ssim, cfg.lambda_scale
    )
    evaluator = Evaluator(cfg.eval_render_num)

    progress_bar = tqdm.tqdm(
        total=cfg.num_iterations, ncols=120, postfix={"loss": float("inf")}
    )
    tb_writer = SummaryWriter(Path(cfg.output_path) / "tensorboard")
    step = 0
    for data in train_dataloader:
        step += 1
        all_tb_info: Dict[str, Any] = {}

        data_to_device(data)
        model_output = gaussian_model(data)
        loss_dict = loss_computer.get_loss_dict(
            model_output["render_img"],
            data["image"],
            data["mask"],
        )
        loss_dict["total"].backward()

        all_tb_info["train/loss"] = {}
        for name, loss in loss_dict.items():
            all_tb_info["train/loss"][name] = loss.item()

        with torch.no_grad():
            # save model
            if step in cfg.save_model_iterations:
                model_save_path = (
                    Path(cfg.output_path) / "checkpoints" / f"iteration_{step}.pth"
                )
                model_save_path.parent.mkdir(exist_ok=True)
                torch.save(gaussian_model, model_save_path)
            # evaluation
            if step == 1 or step % cfg.eval_every == 0:
                gaussian_model.eval()
                metrics_dict = evaluator(eval_dataloader, gaussian_model)
                for key, value in metrics_dict.items():
                    if "render" in key:
                        all_tb_info[f"render/{key}"] = value
                    if key in ["psnr", "ssim", "lpips"]:
                        all_tb_info[f"eval/{key}"] = value
                gaussian_model.train()
            # refine
            if cfg.refine_start < step <= cfg.refine_stop:
                gaussian_model.update_statistics(data, model_output)
                if (step - cfg.refine_start) % cfg.refine_every == 0:
                    tb_info = gaussian_model.densify_and_prune()
                    all_tb_info.update(tb_info)
                if (step - cfg.refine_start) % cfg.reset_opacities_every == 0:
                    gaussian_model.reset_opacities()
            # increase sh_degree
            if cfg.sh_degree_interval != 0 and step % cfg.sh_degree_interval == 0:
                gaussian_model.up_sh_degree()
            # update learning_rate
            gaussian_model.update_learning_rate(step)
            # write to tensorboard
            if step == 1 or step % cfg.log_every == 0:
                tb_report(tb_writer, step, all_tb_info)
            # update progress_bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    {"loss": "%7.5f" % (loss_dict["total"].item())}
                )
                progress_bar.update(10)

        optimizer.step()
        optimizer.zero_grad()

    progress_bar.update(progress_bar.total - progress_bar.n)
    progress_bar.close()
    tb_writer.close()


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


def tb_report(tb_writer: SummaryWriter, step: int, tb_info: Dict[str, Any]):
    for key, value in tb_info.items():
        if isinstance(value, dict):
            tb_writer.add_scalars(key, value, step, walltime=time.time())
        elif isinstance(value, float) or isinstance(value, int):
            tb_writer.add_scalar(key, value, step, walltime=time.time())
        elif isinstance(value, np.ndarray):
            tb_writer.add_image(
                key, value, step, walltime=time.time(), dataformats="HWC"
            )
        else:
            logger.warning(
                f"unsupported type for tensorboard report: {type(value)} (key={key})"
            )


def parse_cfg(args) -> easydict.EasyDict:
    if not Path(args.config).exists():
        raise FileNotFoundError(f"config does not exist: {args.config}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"data does not exist: {args.data}")

    with open(args.config, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg["data"] = args.data
        cfg["output"] = args.output
    cfg = easydict.EasyDict(cfg)

    if cfg.num_iterations not in cfg.save_model_iterations:
        logger.warning(
            "num_iterations is not in save_model_iterations, append num_iterations to save_model_iterations"
        )
        cfg.save_model_iterations.append(cfg.num_iterations)

    project_name = Path(cfg.data).stem
    time_formatted = datetime.now().strftime(r"%m-%d_%H-%M-%S")
    cfg.output = str(Path(cfg.output) / project_name / time_formatted)
    Path(cfg.output).mkdir(parents=True)
    with open(Path(cfg.output) / "config.yaml", "w") as f:
        yaml.dump(dict(cfg), f, sort_keys=False)

    return cfg


if __name__ == "__main__":
    set_global_state(seed=0, device="cuda:0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()
    cfg = parse_cfg(args)

    train(cfg)
    logger.info("training finished")
    eval(cfg.output)
