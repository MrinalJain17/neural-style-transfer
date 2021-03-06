from typing import List

import numpy as np
import torch
from torchvision import transforms


def get_transform(
    resize: List[int] = [512, 512], normalize: bool = True, gray: bool = False
) -> transforms.Compose:
    transform = transforms.Compose([transforms.Resize(resize)])

    if normalize:
        transform.transforms.append(
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            )
        )

    if gray:
        transform.transforms.append(transforms.Grayscale(3))

    return transform


def denormalize(image: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).type_as(image)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).type_as(image)

    return torch.clamp((image * std) + mean, 0, 1)
