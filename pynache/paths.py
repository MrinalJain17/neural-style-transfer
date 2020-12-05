"""
This file sets the default path values.

It stores:

1. `REPOSITORY_ROOT` - points the to root of the repository.
2. `DEFAULT_IMAGES` - points to the directory where the images (style, content,
etc are stored). It's the directory "images" in the root of the repository.
3. `COCO_ROOT` - points to the directory for COCO images (for Fast Style Transfer)

The code in the entire repository will use these paths by default.
"""

import os
from pathlib import Path


def is_cluster(cluster_type: str = "PRINCE") -> bool:
    """
    Will return True if executed on NYU's HPC Cluster Prince (by default).
    """
    env = os.environ.get("CLUSTER")
    return True if env == cluster_type else False


def _repository_root() -> Path:
    return (Path(__file__).resolve().parents[1]).absolute()


def _storage_root() -> Path:
    path = _repository_root() / "images"
    return path.absolute()


def _coco_root() -> Path:
    path = (
        (Path(os.environ.get("BEEGFS")) / "coco")
        if is_cluster()
        else (_repository_root() / "coco")
    )

    return path.absolute()


REPOSITORY_ROOT: str = _repository_root().as_posix()
DEFAULT_IMAGES: str = _storage_root().as_posix()

# COCO Dataset
COCO_ROOT = _coco_root().as_posix()
COCO_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_DIR = COCO_URL.split("/")[-1].split(".")[0]

# Models
VGG_NORMALIZED_STATE_DICT = "https://github.com/MrinalJain17/vgg-normalized/releases/download/v1.0/vgg19_normalized.pth"
