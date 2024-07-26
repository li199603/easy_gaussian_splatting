import torch
import torch.nn as nn
from torch import Tensor
from scene import Pointcloud
import numpy as np
from .utils import *


class GaussianModel(nn.Module):
    def __init__(
        self,
        pc: Pointcloud,
        sh_degree: int,
        sh_degree_interval: int,
        means_lr_init: float,
        means_lr_final: float,
        means_lr_schedule_max_steps: int,
        densify_grad_thresh: float,
        densify_scale_thresh: float,
        num_splits: int,
        prune_radii_ratio_thresh: float,
        prune_scale_thresh: float,
        min_opacity: float,
        use_scale_regularization: bool,
        max_scale_ratio: float,
    ):
        super().__init__()
        # mean and covariance
        self.means = nn.Parameter(torch.tensor(pc.xyzs, dtype=torch.float32))  # [N, 3]
        dists = k_nearest_neighbors_dists(pc.xyzs, k=3)
        avg_dist = np.mean(dists, axis=1, keepdims=True)
        avg_dist = np.repeat(avg_dist, repeats=3, axis=1)
        scales = torch.tensor(avg_dist, dtype=torch.float32)
        log_scales = torch.log(scales)
        self.log_scales = nn.Parameter(log_scales)  # [N, 3]
        quats = torch.zeros((pc.nbr_points, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)  # [N, 4]  wxyz
        # color
        dim_sh = (sh_degree + 1) ** 2
        shs = torch.zeros((pc.nbr_points, dim_sh, 3), dtype=torch.float32)
        shs[:, 0] = torch.tensor(
            to_sh_on_zero_degree(pc.rgbs / 255.0), dtype=torch.float32
        )
        self.sh_0 = nn.Parameter(shs[:, 0:1])  # [N, 1, 3]
        self.sh_rest = nn.Parameter(shs[:, 1:])  # [N, dim_sh - 1, 3]
        # opacity
        opacities = 0.8 * torch.ones((pc.nbr_points,), dtype=torch.float32)
        logit_opacities = torch.logit(opacities)
        self.logit_opacities = nn.Parameter(logit_opacities)  # [N, ]
        # statistics for densify and prune
        self.grad_norm_accum = torch.zeros(
            (pc.nbr_points,), dtype=torch.float32, device="cuda"
        )  # [N, ]
        self.collecting_counts = torch.zeros(
            (pc.nbr_points,), dtype=torch.float32, device="cuda"
        )  # [N, ]
        self.max_radii = torch.zeros(
            (pc.nbr_points,), dtype=torch.float32, device="cuda"
        )  # [N, ]

        self.optimizer = None
        self.active_sh_degree = 0 if sh_degree_interval != 0 else sh_degree
        self.means_lr_scheduler = LR_Scheduler(
            means_lr_init, means_lr_final, means_lr_schedule_max_steps
        )
        self.MAX_SH_DEGREE = sh_degree
        self.DENSIFY_GRAD_THRESH = densify_grad_thresh
        self.DENSIFY_SCALE_THRESH = densify_scale_thresh
        self.NUM_SPLITS = num_splits
        self.PRUNE_RADII_THRESH = prune_radii_ratio_thresh
        self.PRUNE_SCALE_THRESH = prune_scale_thresh
        self.MIN_OPACITY = min_opacity
        self.USE_SCALE_REGULARIZATION = use_scale_regularization
        self.MAX_SCALE_RARIO = torch.tensor(
            max_scale_ratio, dtype=torch.float32, device="cuda"
        )

        self.cuda()
