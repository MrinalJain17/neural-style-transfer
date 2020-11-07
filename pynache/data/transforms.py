from typing import List

import numpy as np
import torch
from torchvision import transforms

_MUL = 255.0


def get_transform(
    resize: List[int] = [512, 512], normalize: bool = False
) -> transforms.Compose:
    transform = transforms.Compose([transforms.Resize(resize)])

    if normalize:
        transform.transforms.append(
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]) * _MUL,
                std=np.array([1.0, 1.0, 1.0]),
            )
        )

    return transform


def denormalize(image: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).type_as(image) * _MUL
    std = torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1).type_as(image)

    return torch.clamp((image * std) + mean, 0, _MUL)
