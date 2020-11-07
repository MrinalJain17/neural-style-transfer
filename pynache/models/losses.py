from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STYLE_WEIGHING_SCHEMES = ["default", "improved"]


def gram_matrix(feature_maps: torch.Tensor, shift: int = 0) -> torch.Tensor:
    gram = torch.einsum("bxhw,byhw->bxy", feature_maps - shift, feature_maps - shift)
    return gram  # Shape: (B, C, C)


def gram_chain(feature_maps: Tuple[torch.Tensor, torch.Tensor], shift: int = 0):
    gram = torch.einsum(
        "bxhw,byhw->bxy", feature_maps[0] - shift, feature_maps[1] - shift
    )
    return gram  # Shape: (B, C1, C2)


def _compute_weights(num_layers, weight_type="default"):
    assert weight_type in STYLE_WEIGHING_SCHEMES, "Invalid weighing scheme passed"
    if weight_type == "default":
        weights = [1.0 for _ in range(num_layers)]
    else:
        weights = [2 ** (num_layers - (i + 1)) for i in range(num_layers)]

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
            B, C, H, W = input.size()
            G, A = (
                gram_matrix(input, self.activation_shift) / (C * C * H * W),
                gram_matrix(target, self.activation_shift) / (C * C * H * W),
            )
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum") / B)

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
            B, C1, H, W = arr1.size()
            _, C2, _, _ = arr2.size()

            arr2 = F.interpolate(arr2, size=(H, W))
            return gram_chain((arr1, arr2), shift=self.activation_shift) / (
                C1 * C2 * H * W
            )  # Shape: (B, C1, C2)

        for idx in range(self.num_layers - 1):
            input_grams.append(_compute_gram(inputs[idx], inputs[idx + 1]))
            target_grams.append(_compute_gram(targets[idx], targets[idx + 1]))

        loss_list = []
        for (G, A, weight) in zip(input_grams, target_grams, self.weights):
            B, _, _ = G.size()
            loss_list.append(weight * F.mse_loss(G, A, reduction="sum") / B)

        return sum(loss_list)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert (len(inputs) == 1) and (len(targets) == 1)

        input, target = inputs[0], targets[0]
        return F.mse_loss(input, target)
