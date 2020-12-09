from pathlib import Path

from pynache.data.transforms import get_transform
from pynache.paths import COCO_DIR, COCO_ROOT
from torch.utils.data import Dataset
from torchvision.io import read_image


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
        image = read_image(self.image_paths[index]).float().div(255)
        if image.shape[0] == 1:  # Some images in COCO are gray scale
            image = image.repeat(3, 1, 1)

        transform = get_transform(resize=[256, 256], normalize=True, gray=False)
        return transform(image)


def get_coco():
    path = (Path(COCO_ROOT) / COCO_DIR).as_posix()
    return COCODataset(path)
