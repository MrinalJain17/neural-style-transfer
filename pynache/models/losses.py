from typing import List

import torch
import torch.nn as nn

STYLE_WEIGHING_SCHEMES = ["default", "improved"]


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    gram = torch.einsum("bxhw,byhw->bxy", feature_maps, feature_maps)
    return gram  # Shape: (B, C, C)


class StyleLoss(nn.Module):
    def __init__(self, num_layers, weights="default"):
        super(StyleLoss, self).__init__()
        assert weights in STYLE_WEIGHING_SCHEMES, "Invalid weighing scheme passed"
        if weights == "default":
            self.weights = [1.0 for _ in range(num_layers)]
        else:
            self.weights = [2 ** (num_layers - (i + 1)) for i in range(num_layers)]

        sum_ = sum(self.weights)
        self.weights = [w / sum_ for w in self.weights]

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.weights)

        loss_list = []
        for (input, target, weight) in zip(inputs, targets, self.weights):
            B, C, H, W = input.size()
            denominator = B * 4.0 * (C ** 2) * ((H * W) ** 2)
            G, A = gram_matrix(input), gram_matrix(target)
            loss_list.append(weight * (torch.sum(torch.square(G - A)) / denominator))

        return sum(loss_list)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        assert (len(inputs) == 1) and (len(targets) == 1)

        input, target = inputs[0], targets[0]
        B, C, H, W = input.size()
        denominator = B * 2.0

        return torch.sum(torch.square(input - target)) / denominator
