from typing import Tuple

import numpy as np
from torchvision import transforms


def get_transform(
    resize: Tuple[int, int] = (512, 512), normalize: bool = False
) -> transforms.Compose:
    transform = transforms.Compose([transforms.Resize(resize)])

    if normalize:
        transform.transforms.append(
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]) * 255,
                std=np.array([0.229, 0.224, 0.225]) * 255,
            )
        )

    return transform
