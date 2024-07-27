from typing import Literal, Tuple, List, Dict, Any, Optional
import random
from .colmap_loader import load_colmap_data
from .blender_loader import load_blender_data
from torch.utils.data import Dataset
from pathlib import Path
import json


def get_dataset_indexes(
    num: int, eval: bool, eval_split_ratio: float
) -> Tuple[List[int], List[int]]:
    indexes = list(range(num))
    random.shuffle(indexes)
    split_point = int(num * eval_split_ratio)
    eval_indexes = indexes[:split_point]
    train_indexes = indexes[split_point:] if eval else indexes
    return train_indexes, eval_indexes


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
        num_iterations: int,
        eval: bool,
        eval_split_ratio: float,
        use_masks: bool,
    ):
        if data_format == "colmap":
            self.frames, self.pc = load_colmap_data(data_path, use_masks)
        elif data_format == "blender":
            self.frames, self.pc = load_blender_data(data_path, use_masks)
        else:
            raise ValueError(f"Invalid data_format: {data_format}")
        self.train_indexes, self.eval_indexes = get_dataset_indexes(
            len(self.frames), eval, eval_split_ratio
        )
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
        return frame.to_data()

    def _export_cameras_json(self, save_path: Path):
        frame_jsons = [frame.to_json(id) for id, frame in enumerate(self.frames)]
        with open(save_path, "w") as f:
            json.dump(frame_jsons, f)
