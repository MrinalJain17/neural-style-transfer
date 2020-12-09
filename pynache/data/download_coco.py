from pathlib import Path

from pynache.paths import COCO_DIR, COCO_ROOT, COCO_URL
from torchvision.datasets.utils import download_and_extract_archive

if __name__ == "__main__":
    if not (Path(COCO_ROOT) / COCO_DIR).is_dir():
        download_and_extract_archive(
            url=COCO_URL, download_root=COCO_ROOT, remove_finished=True
        )
