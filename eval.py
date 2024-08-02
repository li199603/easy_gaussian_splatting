import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import Tensor
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from typing import Dict, Any, Optional
import random
from pathlib import Path
from scene import Scene, data_to_device
import yaml
import easydict  # type: ignore
from model import gaussian
from loguru import logger
from utils import set_global_state, load_gaussian_model
import time


class Evaluator:
    def __init__(self, eval_render_num: int) -> None:
        self.eval_render_num = eval_render_num
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()

    @torch.no_grad()
    def __call__(self, dataloader: DataLoader, model: nn.Module) -> Dict[str, Any]:
        metrics_dict: Dict[str, Any] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        render_indexes = list(range(len(dataloader)))
        if len(render_indexes) > self.eval_render_num:
            render_indexes = random.sample(render_indexes, k=self.eval_render_num)

        render_count = 0
        cost = 0.0
        for i, data in enumerate(dataloader):
            data_to_device(data, non_blocking=False)
            t0 = time.time()
            model_output = model(data)
            t1 = time.time()
            cost += t1 - t0

            gt_img: Tensor = data["image"]
            mask: Tensor = data["mask"]
            render_img: Tensor = model_output["render_img"]

            mask = mask.unsqueeze(2).repeat(1, 1, 3)
            render_img = mask * gt_img + (1.0 - mask) * render_img
            render_img = render_img.permute(2, 0, 1)[None, ...]  # [1, 3, H, W]
            gt_img = gt_img.permute(2, 0, 1)[None, ...]  # [1, 3, H, W]

            metrics_dict["psnr"] += self.psnr(gt_img, render_img).item()
            metrics_dict["ssim"] += self.ssim(gt_img, render_img).item()
            metrics_dict["lpips"] += self.lpips(gt_img, render_img).item()

            if i in render_indexes:
                render_count += 1
                render_img = (
                    torch.cat((data["image"], model_output["render_img"]), dim=1)
                    .cpu()
                    .numpy()
                )
                metrics_dict[f"render_{render_count}"] = render_img

        metrics_dict["psnr"] /= len(dataloader)
        metrics_dict["ssim"] /= len(dataloader)
        metrics_dict["lpips"] /= len(dataloader)
        metrics_dict["fps"] = len(dataloader) / cost

        torch.cuda.empty_cache()
        return metrics_dict


def eval(training_output_path: str, iterations: Optional[int]):
    with open(Path(training_output_path) / "config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = easydict.EasyDict(cfg)
    set_global_state(cfg.random_seed, cfg.device)

    cfg.output = None
    cfg.eval_render_num = 0

    scene = Scene(
        cfg.data,
        cfg.data_format,
        cfg.output,
        cfg.white_background,
        cfg.num_iterations,
        cfg.eval,
        cfg.eval_split_ratio,
        cfg.eval_in_val,
        cfg.eval_in_test,
        cfg.use_masks,
    )
    scene.train_indexes = list(set(scene.train_indexes))
    train_dataloader = DataLoader(
        scene.train_dataset,
        batch_size=1,
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
    )

    gaussian_model: gaussian.GaussianModel = load_gaussian_model(
        Path(training_output_path), iterations
    ).eval()  # type: ignore
    logger.info(f"nbr_gaussians: {gaussian_model.nbr_gaussians}")
    evaluator = Evaluator(cfg.eval_render_num)
    for set_name, dataloder in zip(
        ["train set", "eval set"], [train_dataloader, eval_dataloader]
    ):
        if len(dataloder) == 0:
            logger.info(f"{set_name} is empty, skip evaluation")
            continue
        metrics_dict = evaluator(dataloder, gaussian_model)
        psnr = metrics_dict["psnr"]
        ssim = metrics_dict["ssim"]
        lpips = metrics_dict["lpips"]
        fps = metrics_dict["fps"]
        logger.info(
            f"evaluation in {set_name:>10s}: psnr={psnr:6.3f}, ssim={ssim:6.3f}, lpips={lpips:6.3f}, fps={fps:6.3f}"
        )


if __name__ == "__main__":
    import sys
    import argparse

    fmt = "<green>{time:MMDD-HH:mm:ss.SSSSSS}</green> | <level>{level:5}</level> | <level>{message}</level>"
    level = "DEBUG"
    log_config = {
        "handlers": [
            {"sink": sys.stdout, "format": fmt, "level": level, "enqueue": True}
        ]
    }
    logger.configure(**log_config)  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True)
    parser.add_argument("--iterations", "-i", type=int, default=None)
    args = parser.parse_args()

    eval(args.path, args.iterations)
