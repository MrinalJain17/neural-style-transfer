import torch
from pynache.utils import to_numpy, to_tensor
from skimage.color import rgb2yiq, yiq2rgb


def _add_color(content: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    assert content.ndim == 4, "Expected input of shape (1, C, H, W)"
    assert generated.ndim == 4, "Expected input of shape (1, C, H, W)"

    content, generated = (
        to_numpy(content[0].detach().cpu()),
        to_numpy(generated[0].detach().cpu()),
    )
    generated[:, :, 1:] = rgb2yiq(content)[:, :, 1:]
    generated = yiq2rgb(generated)

    return to_tensor(generated).unsqueeze(0)
