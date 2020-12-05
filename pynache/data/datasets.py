from pathlib import Path

from pynache.data.load_inputs import _load
from pynache.paths import COCO_DIR, COCO_ROOT
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path).absolute()

        self.image_paths = []
        for image in self.path.iterdir():
            self.image_paths.append(image.as_posix())
        self.image_paths.sort()  # To get same order on Windows and Linux (cluster)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        return _load(
            self.image_paths[index], resize=[256, 256], normalize=True, gray=False
        )


def get_coco():
    path = (Path(COCO_ROOT) / COCO_DIR).as_posix()
    return COCODataset(path)
