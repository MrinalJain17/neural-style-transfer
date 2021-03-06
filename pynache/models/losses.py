"""Different components comprising the perceptual loss.

Note that the loss functions do not normalize with respect to the batch size.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_WEIGHING_SCHEMES = ["default", "improved", "recommended"]
_NUM_FILTERS = {
    5: [64, 128, 256, 512, 512],
    16: [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}


def gram_matrix(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    # Shape: (B, C, C) or (B, C1, C2)
    return torch.einsum("bxhw,byhw->bxy", arr1, arr2)


_sample = torch.rand((1, 128, 256, 256))
gram_matrix = torch.jit.trace(gram_matrix, (_sample, _sample.clone()))


class StyleLoss(nn.Module):
    def __init__(self, num_layers, weights="default", activation_shift=False):
        super(StyleLoss, self).__init__()
        self.activation_shift = int(activation_shift)  # Either 0 or 1
        self.weights = _compute_weights(
            num_layers=num_layers, weight_type=weights, chained_gram=False
        )

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.weights)

        loss_list = []
        for (input, target, weight) in zip(inputs, targets, self.weights):
            _, C, H, W = input.size()
            denom = 2 * C * H * W
            input, target = (
                input - self.activation_shift,
                target - self.activation_shift,
            )
            G, A = (
                gram_matrix(input, input) / denom,
                gram_matrix(target, target) / denom,
            )
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum"))

        return torch.stack(loss_list).sum()


class StyleLossChained(nn.Module):
    def __init__(self, num_layers, weights="default", activation_shift=False):
        super(StyleLossChained, self).__init__()
        self.num_layers = num_layers
        self.activation_shift = int(activation_shift)  # Either 0 or 1
        self.weights = _compute_weights(
            num_layers=num_layers, weight_type=weights, chained_gram=True
        )

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.weights) + 1

        input_grams = []
        target_grams = []

        for idx in range(self.num_layers - 1):
            input_grams.append(
                _compute_chained_gram(
                    inputs[idx], inputs[idx + 1], self.activation_shift
                )
            )
            target_grams.append(
                _compute_chained_gram(
                    targets[idx], targets[idx + 1], self.activation_shift
                )
            )

        loss_list = []
        for (G, A, weight) in zip(input_grams, target_grams, self.weights):
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum"))

        return torch.stack(loss_list).sum()


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert (len(inputs) == 1) and (len(targets) == 1)

        input, target = inputs[0], targets[0]
        _, C, H, W = input.size()
        return F.mse_loss(input, target, reduction="sum") / (2 * C * H * W)


class TotalVariation(nn.Module):
    """Total Variation Loss

    Code adapted from the implementation in the library Kornia:
    (https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.TotalVariation)
    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, input: torch.Tensor):
        assert input.ndim == 4, "Expected input of shape (B, C, H, W)"
        pixel_dif1 = (input[..., 1:, :] - input[..., :-1, :]).abs()
        pixel_dif2 = (input[..., :, 1:] - input[..., :, :-1]).abs()

        return pixel_dif1.sum() + pixel_dif2.sum()


@torch.jit.script
def _compute_chained_gram(arr1: torch.Tensor, arr2: torch.Tensor, shift: int):
    _, C1, H, W = arr1.size()
    _, C2, _, _ = arr2.size()
    denom = 2 * ((C1 * C2) ** 0.5) * H * W

    arr1 = arr1 - shift
    arr2 = F.interpolate(arr2, size=(H, W)) - shift
    return gram_matrix(arr1, arr2) / denom


def _compute_weights(num_layers, weight_type="default", chained_gram=False):
    assert weight_type in STYLE_WEIGHING_SCHEMES, "Invalid weighing scheme passed"

    if weight_type in ["default", "improved"]:
        num_layers = (num_layers - 1) if chained_gram else num_layers
        weights = (
            [1.0 for _ in range(num_layers)]
            if weight_type == "default"
            else [2 ** (num_layers - (i + 1)) for i in range(num_layers)]
        )
    else:
        filters = (
            zip(_NUM_FILTERS[num_layers], _NUM_FILTERS[num_layers][1:])
            if chained_gram
            else zip(_NUM_FILTERS[num_layers], _NUM_FILTERS[num_layers])
        )
        weights = [1.0 / (f1 * f2) for (f1, f2) in filters]

    return [w / sum(weights) for w in weights]
