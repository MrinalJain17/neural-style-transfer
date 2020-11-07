from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from pynache.paths import VGG_NORMALIZED_STATE_DICT
from torchvision import models

VGG_LAYER_MAP: Dict[int, str] = {
    1: "conv1_1",
    3: "conv1_2",
    6: "conv2_1",
    8: "conv2_2",
    11: "conv3_1",
    13: "conv3_2",
    15: "conv3_3",
    17: "conv3_4",
    20: "conv4_1",
    22: "conv4_2",
    24: "conv4_3",
    26: "conv4_4",
    29: "conv5_1",
    31: "conv5_2",
    33: "conv5_3",
    35: "conv5_4",
}

FEATURES_CONFIG: Dict = {
    "default": {
        "style": ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        "content": ["conv4_2"],
    },
    "improved": {"style": list(VGG_LAYER_MAP.values()), "content": ["conv4_2"]},
    "recommended": {"style": list(VGG_LAYER_MAP.values()), "content": ["conv5_2"]},
}


class VGGFeatures(nn.Module):
    """Base class for extracting feature maps from intermediate VGG layers.

    Parameters
    ----------

    config : str
        Indicates the intermediate layers in the VGG network for which the activations
        would be stored.

        The format is described in the dictionary `FEATURES_CONFIG`.
    """

    def __init__(
        self,
        config: str = "default",
        use_avg_pool: bool = False,
        use_normalized_vgg: bool = False,
    ) -> None:
        super(VGGFeatures, self).__init__()
        self.model = self._load_vgg(use_normalized_vgg)
        self.use_avg_pool = use_avg_pool

        assert config in FEATURES_CONFIG.keys(), "Invalid configuration passed"
        self.style_layers = FEATURES_CONFIG[config]["style"]
        self.content_layers = FEATURES_CONFIG[config]["content"]

        for layer_num, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d) and self.use_avg_pool:
                self.model[layer_num] = nn.AvgPool2d(kernel_size=2, stride=2)

        self._freeze()

    def _load_vgg(self, use_normalized_vgg):
        model = models.vgg19(pretrained=True)
        if use_normalized_vgg:
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(VGG_NORMALIZED_STATE_DICT)
            )
        return model.features

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        style_features = []
        content_features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            current_layer = VGG_LAYER_MAP.get(layer_num, None)

            if current_layer and (current_layer in self.style_layers):
                style_features.append(x)

            if current_layer and (current_layer in self.content_layers):
                content_features.append(x)

        return style_features, content_features

    def _freeze(self) -> None:
        """Freezes the model parameters and sets it to evaluation mode"""
        for param in self.parameters():
            param.requires_grad = False

        self.eval()
