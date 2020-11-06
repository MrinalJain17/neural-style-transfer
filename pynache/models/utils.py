import torch


class AverageMeter(object):
    """Helper class to track the running average."""

    def __init__(self):
        self.contents = []
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.contents.clear()

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.contents.extend([value] * n)

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0

    @property
    def last(self):
        if self.count > 0:
            return self.contents[-1]
        else:
            return 0


def denormalize(image: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).type_as(image)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).type_as(image)

    return torch.clamp((image * std) + mean, 0, 1)
