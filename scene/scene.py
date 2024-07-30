from typing import Literal, Tuple, List, Dict, Any, Optional
import random
from .colmap_loader import load_colmap_data
from .blender_loader import load_blender_data
from torch.utils.data import Dataset
from pathlib import Path
import json


class SceneDataset(Dataset):
    def __init__(self, scene: "Scene", split: Literal["train", "eval"]):
        super().__init__()
        self.scene = scene
        self.split = split

    def __len__(self):
        return self.scene.nbr_data(self.split)

    def __getitem__(self, idx):
        return self.scene.get_data(self.split, idx)


class Scene:
    def __init__(
        self,
        data_path: str,
        data_format: Literal["colmap", "blender"],
        output_path: Optional[str],
        white_background: bool,
        num_iterations: int,
        eval: bool,
        eval_split_ratio: float,
        eval_in_val: bool,
        eval_in_test: bool,
        use_masks: bool,
    ):
        self.white_background = white_background
        if data_format == "colmap":
            colmap_data = load_colmap_data(data_path, use_masks, eval, eval_split_ratio)
            self.frames, self.pc, self.train_indexes, self.eval_indexes = colmap_data
        elif data_format == "blender":
            blender_data = load_blender_data(data_path, use_masks, eval, eval_in_val, eval_in_test)
            self.frames, self.pc, self.train_indexes, self.eval_indexes = blender_data
        else:
            raise ValueError(f"Invalid data_format: {data_format}")

        if num_iterations < len(self.train_indexes):
            raise ValueError(
                "the number of iterations is less than the number of training data"
            )
        self.train_indexes *= num_iterations // len(self.train_indexes) + 1
        self.train_indexes = self.train_indexes[:num_iterations]
        self.train_dataset = SceneDataset(self, "train")
        self.eval_dataset = SceneDataset(self, "eval")

        if output_path is not None:
            self._export_cameras_json(Path(output_path) / "cameras.json")

    def nbr_data(self, split: Literal["train", "eval"]) -> int:
        if split == "train":
            return len(self.train_indexes)
        elif split == "eval":
            return len(self.eval_indexes)
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_data(self, split: Literal["train", "eval"], index: int) -> Dict[str, Any]:
        if split == "train":
            frame = self.frames[self.train_indexes[index]]
        elif split == "eval":
            frame = self.frames[self.eval_indexes[index]]
        else:
            raise ValueError(f"Invalid split: {split}")
        return frame.to_data(self.white_background)

    def _export_cameras_json(self, save_path: Path):
        frame_jsons = [frame.to_json(id) for id, frame in enumerate(self.frames)]
        with open(save_path, "w") as f:
            json.dump(frame_jsons, f)
