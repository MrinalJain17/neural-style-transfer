import numpy as np
import torch


def to_numpy(arr: torch.Tensor) -> np.ndarray:
    return arr.permute((1, 2, 0)).numpy()


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute((2, 0, 1))
