from pathlib import Path
from typing import Tuple

import torch
from pynache.data.transforms import get_transform
from pynache.paths import DEFAULT_IMAGES
from torchvision.io import read_image

STYLE_ROOT: Path = Path(DEFAULT_IMAGES) / "inputs" / "style"
CONTENT_ROOT: Path = Path(DEFAULT_IMAGES) / "inputs" / "content"


def _load(path: str, resize: Tuple[int, int], normalize: bool) -> torch.Tensor:
    image = read_image(path).float().div(255)

    if resize is None:
        return image

    transform = get_transform(resize=resize, normalize=normalize)
    return transform(image)


def load_style(
    name: str, resize: Tuple[int, int] = (512, 512), normalize: bool = False
) -> torch.Tensor:
    path = list(STYLE_ROOT.glob(f"{name}.*"))
    assert len(path) == 1, f"Style image '{name}' does not exist"
    path = path[0].as_posix()

    return _load(path, resize=resize, normalize=normalize)


def load_content(
    name: str, resize: Tuple[int, int] = (512, 512), normalize: bool = False
) -> torch.Tensor:
    path = list(CONTENT_ROOT.glob(f"{name}.*"))
    assert len(path) == 1, f"Content image '{name}' does not exist"
    path = path[0].as_posix()

    return _load(path, resize=resize, normalize=normalize)
