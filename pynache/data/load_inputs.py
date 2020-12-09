from pathlib import Path
from typing import List

from pynache.data.transforms import get_transform
from pynache.paths import DEFAULT_IMAGES
import torch
from torchvision.io import read_image

STYLE_ROOT: Path = Path(DEFAULT_IMAGES) / "inputs" / "style"
CONTENT_ROOT: Path = Path(DEFAULT_IMAGES) / "inputs" / "content"


def _load(path: str, resize: List[int], normalize: bool, gray: bool) -> torch.Tensor:
    image = read_image(path).float().div(255)
    transform = get_transform(resize=resize, normalize=normalize, gray=gray)

    return transform(image)


def load_style(
    name: str,
    resize: List[int] = [512, 512],
    normalize: bool = True,
    gray: bool = False,
) -> torch.Tensor:
    path = list(STYLE_ROOT.glob(f"{name}.*"))
    assert len(path) == 1, f"Style image '{name}' does not exist"
    path = path[0].as_posix()

    return _load(path, resize=resize, normalize=normalize, gray=gray)


def load_content(
    name: str,
    resize: List[int] = [512, 512],
    normalize: bool = True,
    gray: bool = False,
) -> torch.Tensor:
    path = list(CONTENT_ROOT.glob(f"{name}.*"))
    assert len(path) == 1, f"Content image '{name}' does not exist"
    path = path[0].as_posix()

    return _load(path, resize=resize, normalize=normalize, gray=gray)
