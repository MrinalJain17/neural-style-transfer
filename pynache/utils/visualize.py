from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pynache.utils import to_numpy
import torch


def plot_tensor(
    tensor: Union[torch.Tensor, np.ndarray], figsize: Tuple[int, int] = (8, 8)
):
    if isinstance(tensor, torch.Tensor):
        tensor = to_numpy(tensor)

    assert tensor.shape[-1] == 3, "Expected an image in channel-last format"

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(tensor)
    return ax
