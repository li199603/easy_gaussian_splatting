import argparse
import yaml
import easydict  # type: ignore
from pathlib import Path
from loguru import logger
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
from utils import set_global_state, load_camera_states
from viewer import Viewer, CameraState


def train(cfg: easydict.EasyDict):
    scene = Scene(
        cfg.data,
        cfg.data_format,
        cfg.output,
        cfg.total_iterations,
        cfg.eval,
        cfg.eval_split_ratio,
        cfg.eval_in_val,
        cfg.eval_in_test,
        cfg.use_masks,
        cfg.mask_expand_pixels,
        cfg.white_background,
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
        cfg.white_background,
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
    tb_path = Path(cfg.output) / "tensorboard"
    logger.info(f"monitor training status: tensorboard --logdir {tb_path}")
    tb_writer = SummaryWriter(tb_path)
    viewer = None
    if cfg.view_online:
        viewer = construct_viewer(gaussian_model, Path(cfg.output))

    progress_bar = tqdm.tqdm(
        total=cfg.total_iterations, ncols=120, postfix={"loss": float("inf")}
    )
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
                    Path(cfg.output) / "checkpoints" / f"iterations_{step}.pth"
                )
                model_save_path.parent.mkdir(exist_ok=True)
                torch.save(gaussian_model, model_save_path)
            # evaluation
            if len(eval_dataloader) != 0 and (step == 1 or step % cfg.eval_every == 0):
                gaussian_model.eval()
                metrics_dict = evaluator(eval_dataloader, gaussian_model)
                for key, value in metrics_dict.items():
                    if "render" in key:
                        all_tb_info[f"render/{key}"] = value
                    if key in ["psnr", "ssim", "lpips", "fps"]:
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
            if (
                step == 1
                or step % cfg.log_every == 0
                or step % cfg.eval_every == 0
                or (step - cfg.refine_start) % cfg.refine_every == 0
            ):
                tb_report(tb_writer, step, all_tb_info)
            # update progress_bar
            if step % 10 == 0:
                progress_bar.set_postfix(
                    {"loss": "%7.5f" % (loss_dict["total"].item())}
                )
                progress_bar.update(10)

        optimizer.step()
        optimizer.zero_grad()

        if viewer is not None:
            viewer.update_render_image()

    progress_bar.update(progress_bar.total - progress_bar.n)
    progress_bar.close()
    tb_writer.close()


def construct_viewer(
    gaussian_model: gaussian.GaussianModel, cameras_json_path: Path
) -> Viewer:
    camera_states = load_camera_states(cameras_json_path)

    @torch.no_grad()
    def gs_render_func(camera_state: CameraState) -> np.ndarray:
        gaussian_model.eval()
        data = {
            "w2c": torch.tensor(camera_state.w2c, dtype=torch.float32, device="cuda"),
            "K": torch.tensor(camera_state.K, dtype=torch.float32, device="cuda"),
            "height": camera_state.height,
            "width": camera_state.width,
        }
        image = gaussian_model(data)["render_img"].cpu().numpy()
        gaussian_model.train()
        return image

    viewer = Viewer(gs_render_func, camera_states, in_training_mode=True)
    return viewer


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
        cfg["view_online"] = args.view_online
    cfg = easydict.EasyDict(cfg)

    project_name = Path(cfg.data).stem
    time_formatted = datetime.now().strftime(r"%m-%d_%H-%M-%S")
    cfg.output = str(Path(cfg.output) / project_name / time_formatted)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="output")
    parser.add_argument("--view_online", action="store_true")
    args = parser.parse_args()
    cfg = parse_cfg(args)
    set_global_state(cfg.random_seed, cfg.device)

    if cfg.total_iterations not in cfg.save_model_iterations:
        logger.warning(
            "total_iterations is not in save_model_iterations, append total_iterations to save_model_iterations"
        )
        cfg.save_model_iterations.append(cfg.total_iterations)

    logger.info(f"output dir: {cfg.output}")
    Path(cfg.output).mkdir(parents=True)
    with open(Path(cfg.output) / "config.yaml", "w") as f:
        yaml.dump(dict(cfg), f, sort_keys=False)

    train(cfg)
    logger.info("training finished")
    logger.info("--------------------- evaluation ---------------------")
    for iteration in cfg.save_model_iterations:
        eval(cfg.output, iteration)
