import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    (B, C, H, W) = feature_maps.size()
    features = feature_maps.view(B, C, W * H)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (C * H * W)
    return gram  # Shape: (B, C, C)


class MSEWrapper(nn.Module):
    def __init__(self, weights=None):
        super(MSEWrapper, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        assert isinstance(inputs, list) and isinstance(targets, list)
        assert len(inputs) == len(targets)

        loss_list = [
            F.mse_loss(input, target) for (input, target) in zip(inputs, targets)
        ]

        if self.weights is None:
            return sum(loss_list) / len(loss_list)
