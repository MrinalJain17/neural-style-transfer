"""
This file sets the default path values.

It stores:

1. `REPOSITORY_ROOT` - points the to root of the repository.
2. `DEFAULT_IMAGES` - points to the directory where the images (style, content,
etc are stored). It's the directory "images" in the root of the repository.

The code in the entire repository will use these paths by default.
"""

from pathlib import Path


def _repository_root() -> Path:
    return (Path(__file__).resolve().parents[1]).absolute()


def _storage_root() -> Path:
    path = _repository_root() / "images"
    return path.absolute()


REPOSITORY_ROOT: str = _repository_root().as_posix()
DEFAULT_IMAGES: str = _storage_root().as_posix()
VGG_NORMALIZED_STATE_DICT = "https://github.com/MrinalJain17/vgg-normalized/releases/download/v1.0/vgg19_normalized.pth"
