from torch.utils.data import DataLoader
from typing import Dict, Any
import torch.nn as nn


class Evaluator:
    def __init__(self) -> None:
        pass

    def __call__(selfself, dataloader: DataLoader, model: nn.Module) -> Dict[str, Any]:
        return {}
