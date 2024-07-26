import torch
import torch.nn as nn
from torch import Tensor
from scene import Pointcloud
import numpy as np
from .utils import *
from typing import List, Optional, Dict, Any
from gsplat.rendering import rasterization  # type: ignore


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

        self.optimizer: Optional[torch.optim.Optimizer] = None
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
        self.MAX_SCALE_RATIO = torch.tensor(
            max_scale_ratio, dtype=torch.float32, device="cuda"
        )

        self.cuda()

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.logit_opacities)

    @property
    def shs(self) -> Tensor:
        return torch.cat([self.sh_0, self.sh_rest], dim=1)

    @property
    def param_names(self) -> List[str]:
        return ["means", "log_scales", "quats", "sh_0", "sh_rest", "logit_opacities"]

    def register_gs_optimizer(self, optimizer: torch.optim.Optimizer):
        if self.optimizer is not None:
            raise RuntimeError("optimizer has been registered")
        self.optimizer = optimizer

    def up_sh_degree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.MAX_SH_DEGREE)

    def update_learning_rate(self, step: int):
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been registered")
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "means":
                param_group["lr"] = self.means_lr_scheduler(step)
                return
        raise RuntimeError("the param_group 'means' isn't in the optimizer")

    def reset_opacities(self):
        max_opacities = torch.full_like(
            self.logit_opacities, self.MIN_OPACITY * 2.0, device="cuda"
        )
        target_opacities = torch.min(self.opacities * 0.5, max_opacities)
        self.logit_opacities = nn.Parameter(torch.logit(target_opacities))
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "logit_opacities":
                param_state = self.optimizer.state[param_group["params"][0]]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                del self.optimizer.state[param_group["params"][0]]
                param_group["params"][0] = self.logit_opacities
                self.optimizer.state[param_group["params"][0]] = param_state
                return
        raise RuntimeError("the param_group 'logit_opacities' isn't in the optimizer")

    def _get_clone_gaussians(self, mask: Tensor) -> Dict[str, Tensor]:
        new_means = self.means[mask]
        new_log_scales = self.log_scales[mask]
        new_quats = self.quats[mask]
        new_sh_0 = self.sh_0[mask]
        new_sh_rest = self.sh_rest[mask]
        new_logit_opacities = self.logit_opacities[mask]
        return {
            "means": new_means,
            "log_scales": new_log_scales,
            "quats": new_quats,
            "sh_0": new_sh_0,
            "sh_rest": new_sh_rest,
            "logit_opacities": new_logit_opacities,
        }

    def _get_split_gaussians(self, mask: Tensor) -> Dict[str, Tensor]:
        total_num_split = int(torch.sum(mask)) * self.NUM_SPLITS
        centered_samples = torch.randn((total_num_split, 3), device="cuda")
        scaled_samples = self.scales[mask].repeat(self.NUM_SPLITS, 1) * centered_samples
        rotmats = quat_to_rotmat(self.quats[mask].repeat(self.NUM_SPLITS, 1))
        # [K, 3, 3] * [K, 3, 1] -> [K, 3, 1] -> [K, 3]
        rotated_samples = torch.bmm(rotmats, scaled_samples.unsqueeze(-1)).squeeze(-1)
        new_means = self.means[mask].repeat(self.NUM_SPLITS, 1) + rotated_samples
        scale_scaling = 0.8 * self.NUM_SPLITS
        new_scales = self.scales[mask].repeat(self.NUM_SPLITS, 1) / scale_scaling
        new_log_scales = torch.log(new_scales)
        new_quats = self.quats[mask].repeat(self.NUM_SPLITS, 1)
        new_sh_0 = self.sh_0[mask].repeat(self.NUM_SPLITS, 1, 1)
        new_sh_rest = self.sh_rest[mask].repeat(self.NUM_SPLITS, 1, 1)
        new_logit_opacities = self.logit_opacities[mask].repeat(self.NUM_SPLITS)
        return {
            "means": new_means,
            "log_scales": new_log_scales,
            "quats": new_quats,
            "sh_0": new_sh_0,
            "sh_rest": new_sh_rest,
            "logit_opacities": new_logit_opacities,
        }

    def update_statistics(self, data: Dict[str, Any], model_output: Dict[str, Tensor]):
        max_hw = max(data["height"], data["width"])
        radii = model_output["batch_radii"].detach()[0] / max_hw
        xys_absgrad = model_output["batch_xys"].absgrad.detach()[0]  # type: ignore

        visible = radii > 0.0  # [N, ]
        self.max_radii[visible] = torch.max(self.max_radii[visible], radii[visible])
        grads = torch.norm(xys_absgrad, dim=-1) * max_hw  # [N, ]
        self.grad_norm_accum[visible] = self.grad_norm_accum[visible] + grads[visible]
        self.collecting_counts[visible] = self.collecting_counts[visible] + 1

    def _densify_in_model_and_optimizer(self, new_gaussians: Dict[str, Tensor]):
        # densify in model
        self.means = torch.nn.Parameter(
            torch.cat([self.means, new_gaussians["means"]], dim=0)
        )
        self.log_scales = torch.nn.Parameter(
            torch.cat([self.log_scales, new_gaussians["log_scales"]], dim=0)
        )
        self.quats = torch.nn.Parameter(
            torch.cat([self.quats, new_gaussians["quats"]], dim=0)
        )
        self.sh_0 = torch.nn.Parameter(
            torch.cat([self.sh_0, new_gaussians["sh_0"]], dim=0)
        )
        self.sh_rest = torch.nn.Parameter(
            torch.cat([self.sh_rest, new_gaussians["sh_rest"]], dim=0)
        )
        self.logit_opacities = torch.nn.Parameter(
            torch.cat([self.logit_opacities, new_gaussians["logit_opacities"]], dim=0)
        )

        # densify in optimizer
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been registered")
        for param_group in self.optimizer.param_groups:
            tensor = new_gaussians[param_group["name"]]
            param_state = self.optimizer.state[param_group["params"][0]]
            param_state["exp_avg"] = torch.cat(
                [param_state["exp_avg"], torch.zeros_like(tensor)], dim=0
            )
            param_state["exp_avg_sq"] = torch.cat(
                [param_state["exp_avg_sq"], torch.zeros_like(tensor)], dim=0
            )

            del self.optimizer.state[param_group["params"][0]]
            param_group["params"][0] = getattr(self, param_group["name"])
            self.optimizer.state[param_group["params"][0]] = param_state

    def _prune_in_model_and_optimizer(self, prune_mask: Tensor):
        # prune in model
        reserve_mask = torch.logical_not(prune_mask)
        self.means = torch.nn.Parameter(self.means[reserve_mask])
        self.log_scales = torch.nn.Parameter(self.log_scales[reserve_mask])
        self.quats = torch.nn.Parameter(self.quats[reserve_mask])
        self.sh_0 = torch.nn.Parameter(self.sh_0[reserve_mask])
        self.sh_rest = torch.nn.Parameter(self.sh_rest[reserve_mask])
        self.logit_opacities = torch.nn.Parameter(self.logit_opacities[reserve_mask])

        # prune in optimizer
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been registered")
        for param_group in self.optimizer.param_groups:
            param_state = self.optimizer.state[param_group["params"][0]]
            param_state["exp_avg"] = param_state["exp_avg"][reserve_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][reserve_mask]

            del self.optimizer.state[param_group["params"][0]]
            param_group["params"][0] = getattr(self, param_group["name"])
            self.optimizer.state[param_group["params"][0]] = param_state

    def densify_and_prune(self) -> Dict[str, Any]:
        # get densify mask
        avg_grad_norm = self.grad_norm_accum / (self.collecting_counts + 1e-8)
        avg_grad_norm[avg_grad_norm.isnan()] = 0.0
        high_grad_mask = avg_grad_norm >= self.DENSIFY_GRAD_THRESH
        split_mask = self.scales.amax(dim=-1) >= self.DENSIFY_SCALE_THRESH
        clone_mask = torch.logical_not(split_mask)
        split_mask = torch.logical_and(split_mask, high_grad_mask)
        clone_mask = torch.logical_and(clone_mask, high_grad_mask)

        # densify
        split_gaussians = None
        if torch.sum(split_mask) != 0:
            split_gaussians = self._get_split_gaussians(split_mask)
        clone_gaussians = None
        if torch.sum(clone_mask) != 0:
            clone_gaussians = self._get_clone_gaussians(clone_mask)
        if split_gaussians is not None or clone_gaussians is not None:
            new_gaussians = {}
            for key in self.param_names:
                tensors = []
                if split_gaussians is not None:
                    tensors.append(split_gaussians[key])
                if clone_gaussians is not None:
                    tensors.append(clone_gaussians[key])
                new_gaussians[key] = torch.cat(tensors, dim=0)
            self._densify_in_model_and_optimizer(new_gaussians)

        # get prune mask
        # densify results in num_zeros_add greater than or equal to zero
        num_zeros_add = self.means.shape[0] - self.max_radii.shape[0]
        if num_zeros_add > 0:
            zeros_tensor = torch.zeros(
                (num_zeros_add,), dtype=torch.float32, device="cuda"
            )
            self.max_radii = torch.cat(
                [
                    self.max_radii,
                    zeros_tensor,
                ],
                dim=0,
            )
            split_mask = torch.cat(
                [
                    split_mask,
                    zeros_tensor,
                ],
                dim=0,
            )
        prune_counts = []
        prune_mask = self.opacities < self.MIN_OPACITY
        prune_counts.append(prune_mask.sum().item())
        prune_mask = torch.logical_or(
            prune_mask, self.max_radii > self.PRUNE_RADII_THRESH
        )
        prune_counts.append(prune_mask.sum().item())
        prune_mask = torch.logical_or(
            prune_mask, self.scales.amax(dim=-1) > self.PRUNE_SCALE_THRESH
        )
        prune_counts.append(prune_mask.sum().item())
        prune_mask = torch.logical_or(prune_mask, split_mask)

        # prune
        if torch.sum(prune_mask) != 0:
            self._prune_in_model_and_optimizer(prune_mask)

        # reset statistics
        num_gaussian = self.means.shape[0]
        self.grad_norm_accum = torch.zeros(
            (num_gaussian,), dtype=torch.float32, device="cuda"
        )
        self.collecting_counts = torch.zeros(
            (num_gaussian,), dtype=torch.float32, device="cuda"
        )
        self.max_radii = torch.zeros(
            (num_gaussian,), dtype=torch.float32, device="cuda"
        )
        torch.cuda.empty_cache()

        tb_info = {
            "train/densify": {
                "split": split_mask.sum().item(),
                "clone": clone_mask.sum().item(),
            },
            "train/prune": {
                "low_opacity": prune_counts[0],
                "large_radii": prune_counts[1] - prune_counts[0],
                "large_scale": prune_counts[2] - prune_counts[1],
            },
            "train/num_gaussian": num_gaussian,
        }
        return tb_info

    def forward(self, data: Dict[str, Any]) -> Dict[str, Optional[Tensor]]:
        w2c = data["w2c"]
        batch_render_imgs, _, meta = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.shs,
            sh_degree=self.active_sh_degree,
            viewmats=w2c[None],
            Ks=data["K"][None],
            width=data["width"],
            height=data["height"],
            absgrad=True,
            packed=False,
        )
        output = {
            "render_img": torch.clamp(batch_render_imgs[0], min=0.0, max=1.0),  # [H, W, C]
            "batch_xys": meta["means2d"],  # [B, N, 2]  B=1
            "batch_radii": meta["radii"],  # [B, N]
        }
        return output

    def get_regularization_dict(self) -> Dict[str, Tensor]:
        regularization_dict = {}
        if self.USE_SCALE_REGULARIZATION:
            scales = self.scales
            scale_reg = torch.max(
                scales.amax(dim=1) / scales.amin(dim=1), self.MAX_SCALE_RATIO
            )
            scale_reg -= self.MAX_SCALE_RATIO
            scale_reg = torch.mean(scale_reg)
            regularization_dict["scale_reg"] = scale_reg
        return regularization_dict
