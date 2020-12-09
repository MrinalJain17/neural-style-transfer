"""This module implements the image transformation network as described in [1].

Differences from original paper:
--------------------------------

- The authors in [1] use "deconvolution" for upsampling. We instead use
interpolation in conjunction with convolutions with the hope to attenuate the
"checkerboard-effect" as explained here - https://distill.pub/2016/deconv-checkerboard/

- Batch normalization is replaced with Instance normalization as described in [2].

- We have removed the final tanh activation because it produced darker images.

References
----------

[1] Johnson J., Alahi A., Fei-Fei L. (2016)
Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016.
ECCV 2016. Lecture Notes in Computer Science, vol 9906. Springer, Cham.
https://doi.org/10.1007/978-3-319-46475-6_43

[2] Ulyanov, D., A. Vedaldi and V. Lempitsky.
Instance Normalization: The Missing Ingredient for Fast Stylization.
ArXiv abs/1607.08022 (2016)

"""
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size // 2),
        bias=False,
        padding_mode="reflect",
    )


def deconv_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        conv_layer(in_channels, out_channels, kernel_size, stride),
    )


def basic_block(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    transpose=False,
    norm=True,
    activation=True,
):
    layer_func = deconv_layer if transpose else conv_layer
    layers = [layer_func(in_channels, out_channels, kernel_size, stride)]

    if norm:
        layers.append(nn.InstanceNorm2d(num_features=out_channels, affine=True))

    if activation:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(ResidualBlock, self).__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels, activation=False)

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        out += identity

        return out


class TransformationNetwork(nn.Module):
    def __init__(self):
        super(TransformationNetwork, self).__init__()
        self.downsample = nn.Sequential(
            basic_block(3, 32, 9, 1),
            basic_block(32, 64, 3, 2),
            basic_block(64, 128, 3, 2),
        )
        self.residual = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.upsample = nn.Sequential(
            basic_block(128, 64, 3, 1, transpose=True),
            basic_block(64, 32, 3, 1, transpose=True),
            basic_block(32, 3, 9, 1, norm=False, activation=False),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)

        return x
