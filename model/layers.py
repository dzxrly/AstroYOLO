from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding: Union[str, int] = 'same',
                 activation=True):
        super(CBM, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Mish()
        self.enable_act = activation

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        if self.enable_act:
            x = self.activation(x)
        return x


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding: Union[str, int] = 'same'):
        super(CBL, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channel):
        super(ResUnit, self).__init__()
        self.cbms = nn.Sequential(
            CBM(in_channel, in_channel, 1),
            CBM(in_channel, in_channel, 3, activation=False)
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(x + self.cbms(x))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, res_unit_number):
        super(CSPLayer, self).__init__()
        hidden_channel = out_channels // 2
        self.cbm_start = CBM(in_channels, hidden_channel, 1)
        self.res_units = nn.Sequential(
            *[ResUnit(hidden_channel) for _ in range(res_unit_number)]
        )
        self.cbm_end = CBM(hidden_channel * 2, out_channels, 1)

    def forward(self, x):
        res_unit = self.res_units(self.cbm_start(x))
        x = torch.cat([self.cbm_start(x), res_unit], dim=1)
        x = self.cbm_end(x)
        return x


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleConv, self).__init__()
        self.down_sample = CBL(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down_sample(x)


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSampleConv, self).__init__()
        self.up_sample = nn.Sequential(
            CBL(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=scale_factor)
        )

    def forward(self, x):
        return self.up_sample(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=None):
        super(SpatialPyramidPooling, self).__init__()
        if pool_sizes is None:
            pool_sizes = [5, 9, 13]
        self.head_conv = nn.Sequential(
            CBL(feature_channels[-1], feature_channels[-1] // 2, 1),
            CBL(feature_channels[-1] // 2, feature_channels[-1], 3),
            CBL(feature_channels[-1], feature_channels[-1] // 2, 1),
        )
        self.max_pools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )

    def forward(self, x):
        x = self.head_conv(x)
        features = [max_pool(x) for max_pool in self.max_pools]
        features = torch.cat([x] + features, dim=1)

        return features


# ----------------------------------------
# For ConvNeXt
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# For ConvNeXt
class ConvNeXtLayer(nn.Module):
    def __init__(self, in_channels, layer_scale_init_value=1e-6):
        super(ConvNeXtLayer, self).__init__()
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.ln = LayerNorm(in_channels, eps=1e-6)
        self.wide_conv = nn.Linear(in_channels, in_channels * 4)
        self.gelu = nn.GELU()
        self.narrow_conv = nn.Linear(in_channels * 4, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        inputs = x
        x = self.conv7x7(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.ln(x)
        x = self.wide_conv(x)
        x = self.gelu(x)
        x = self.narrow_conv(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return inputs + x


# For ConvNeXt
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.conv_next_block = nn.ModuleList([
            ConvNeXtLayer(in_channels, layer_scale_init_value) for _ in range(num_block)
        ])
        self.down_sample = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format='channels_first'),
            DownSampleConv(in_channels, out_channels)
        )

    def forward(self, x):
        for conv_next_layer in self.conv_next_block:
            x = conv_next_layer(x)
        x = self.down_sample(x)
        return x
# ----------------------------------------
