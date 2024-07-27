import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any
import torch


class Pointcloud:
    def __init__(self, xyzs: np.ndarray, rgbs: np.ndarray):
        self.xyzs = xyzs  # [N, 3]
        self.rgbs = rgbs  # [N, 3]

    @property
    def nbr_points(self) -> int:
        return self.xyzs.shape[0]

    def show(
        self,
        show_colors: bool = True,
        point_size: int = 3,
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ):
        import open3d as o3d  # type: ignore

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.xyzs)
        if show_colors:
            point_cloud.colors = o3d.utility.Vector3dVector(self.rgbs / 255.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = background_color
        vis.run()
        vis.destroy_window()


class Frame:
    def __init__(
        self,
        image_path: Path,
        mask_path: Optional[Path],
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w2c: np.ndarray,
    ):
        self.image_path = image_path
        self.mask_path = mask_path
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.w2c = w2c

    def to_json(self, id: int):
        c2w = np.linalg.inv(self.w2c)
        json = {
            "id": id,
            "img_name": self.image_path.stem,
            "width": self.width,
            "height": self.height,
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "fx": self.fx,
            "fy": self.fy,
        }
        return json

    def show_image(self):
        import matplotlib.pyplot as plt

        image = Image.open(self.image_path)
        image = np.array(image, dtype=np.uint8)
        plt.imshow(image)
        plt.show()
        plt.close()

    def show_mask(self, alpha: float = 0.6):
        import matplotlib.pyplot as plt

        if self.mask_path is None:
            raise ValueError("mask_path is None")

        image = np.array(Image.open(self.image_path), np.float64)
        mask = Image.open(self.mask_path)
        if mask.mode != "1":
            raise ValueError("only support mask on '1' mode")
        # 1: object to be removed，0: scene to be reconstructed
        one_hot_mask = np.array(mask, np.float64) / 255.0
        mask_color = np.array([86, 156, 214], dtype=np.float64)[None, None]

        alpha_arr = (
            np.ones_like(image, dtype=np.float64) * alpha * one_hot_mask[..., None]
        )
        masked_image = (1 - alpha_arr) * image + alpha_arr * mask_color
        masked_image = masked_image.astype(np.uint8)
        plt.imshow(masked_image)
        plt.show()
        plt.close()

    def to_data(self) -> Dict[str, Any]:
        w2c = torch.tensor(self.w2c, dtype=torch.float32)

        image = Image.open(self.image_path)
        if image.mode != "RGB":
            raise ValueError("only support image on 'RGB' mode")
        image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
        ih, iw = image_tensor.shape[:2]

        if self.mask_path is not None:
            mask = Image.open(self.mask_path)
            if mask.mode != "1":
                raise ValueError("only support mask on '1' mode")
            mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
            if mask_tensor.shape != image_tensor.shape[:2]:
                raise ValueError(
                    f"mask shape [{mask_tensor.shape[0]}, {mask_tensor.shape[1]}] is not equal to image shape [{ih}, {iw}]"
                )
        else:
            mask_tensor = torch.zeros((ih, iw), dtype=torch.float32)

        downscale_factor = get_downscale_factor(self.height, self.width, ih, iw)
        fx = self.fx * downscale_factor
        fy = self.fy * downscale_factor
        cx = self.cx * downscale_factor
        cy = self.cy * downscale_factor
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        data = {
            "K": K,  # [3, 3]
            "height": ih,
            "width": iw,
            "w2c": w2c,  # [4, 4]
            "image": image,  # [h, w, 3]
            "mask": mask,  # [h, w]
        }
        return data


def get_downscale_factor(
    orig_h: int, orig_w: int, target_h: int, target_w: int
) -> float:
    if orig_h == target_h and orig_w == target_w:
        return 1.0
    h_downscale_factor = target_h / orig_h
    w_downscale_factor = target_w / orig_w
    if abs(h_downscale_factor - w_downscale_factor) > 1e-3:
        raise ValueError(
            f"h_downscale_factor ({h_downscale_factor}) and w_downscale_factor ({w_downscale_factor}) are not close"
        )
    downscale_factor = (h_downscale_factor + w_downscale_factor) / 2
    return downscale_factor


def data_to_device(data: Dict[str, Any]):
    data["K"] = data["K"].cuda(non_blocking=True)
    data["w2c"] = data["w2c"].cuda(non_blocking=True)
    data["image"] = data["image"].cuda(non_blocking=True)
    data["mask"] = data["mask"].cuda(non_blocking=True)
