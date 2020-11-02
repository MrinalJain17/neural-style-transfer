from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_tensor(
    tensor: Union[torch.Tensor, np.ndarray], figsize: Tuple[int, int] = (8, 8)
):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()

    if tensor.shape[0] == 3:  # Channel-first format
        tensor = np.transpose(tensor, (1, 2, 0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(tensor)
    return ax
