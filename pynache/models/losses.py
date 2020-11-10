"""Different components comprising the perceptual loss

Note that the loss functions do not normalize with respect to the batch size.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_WEIGHING_SCHEMES = ["default", "improved"]


def gram_matrix(
    feature_maps: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], shift: int = 0
) -> torch.Tensor:
    if not isinstance(feature_maps, tuple):
        feature_maps = (feature_maps, feature_maps)

    return torch.einsum(
        "bxhw,byhw->bxy", feature_maps[0] - shift, feature_maps[1] - shift
    )  # Shape: (B, C, C) or (B, C1, C2)


def _compute_weights(num_layers, weight_type="default"):
    assert weight_type in STYLE_WEIGHING_SCHEMES, "Invalid weighing scheme passed"
    weights = (
        [1.0 for _ in range(num_layers)]
        if weight_type == "default"
        else [2 ** (num_layers - (i + 1)) for i in range(num_layers)]
    )

    sum_ = sum(weights)
    return [w / sum_ for w in weights]


class StyleLoss(nn.Module):
    def __init__(self, num_layers, weights="default", activation_shift=False):
        super(StyleLoss, self).__init__()
        self.activation_shift = int(activation_shift)  # Either 0 or 1
        self.weights = _compute_weights(num_layers=num_layers, weight_type=weights)

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.weights)

        loss_list = []
        for (input, target, weight) in zip(inputs, targets, self.weights):
            _, C, H, W = input.size()
            denom = 2 * C * H * W
            G, A = (
                gram_matrix(input, shift=self.activation_shift) / denom,
                gram_matrix(target, shift=self.activation_shift) / denom,
            )
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum"))

        return sum(loss_list)


class StyleLossChained(nn.Module):
    def __init__(self, num_layers, weights="default", activation_shift=False):
        super(StyleLossChained, self).__init__()
        self.num_layers = num_layers
        self.activation_shift = int(activation_shift)  # Either 0 or 1
        self.weights = _compute_weights(
            num_layers=num_layers - 1, weight_type=weights
        )  # Because of chaining, we have (num_layers - 1) gram matrices

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.weights) + 1

        input_grams = []
        target_grams = []

        def _compute_gram(arr1, arr2):
            _, C1, H, W = arr1.size()
            _, C2, _, _ = arr2.size()
            denom = 2 * ((C1 * C2) ** 0.5) * H * W

            arr2 = F.interpolate(arr2, size=(H, W))
            return gram_matrix((arr1, arr2), shift=self.activation_shift) / denom

        for idx in range(self.num_layers - 1):
            input_grams.append(_compute_gram(inputs[idx], inputs[idx + 1]))
            target_grams.append(_compute_gram(targets[idx], targets[idx + 1]))

        loss_list = []
        for (G, A, weight) in zip(input_grams, target_grams, self.weights):
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum"))

        return sum(loss_list)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert (len(inputs) == 1) and (len(targets) == 1)

        input, target = inputs[0], targets[0]
        return F.mse_loss(input, target, reduction="sum") / 2.0


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
