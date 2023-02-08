import torch.nn as nn

from model.layers import ConvNeXtBlock, CBM


class ConvNeXt(nn.Module):
    def __init__(self, stem_channels=32, feature_channels=None, num_features=3, num_blocks=None):
        super(ConvNeXt, self).__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3, 9, 3]
        if feature_channels is None:
            feature_channels = [64, 128, 256, 512, 1024]
        self.stem_conv = CBM(3, stem_channels, 3)
        self.stage_1 = ConvNeXtBlock(stem_channels, feature_channels[0], num_blocks[0])
        self.stage_2 = ConvNeXtBlock(feature_channels[0], feature_channels[1], num_blocks[1])
        self.stage_3 = ConvNeXtBlock(feature_channels[1], feature_channels[2], num_blocks[2])
        self.stage_4 = ConvNeXtBlock(feature_channels[2], feature_channels[3], num_blocks[3])
        self.stage_5 = ConvNeXtBlock(feature_channels[3], feature_channels[4], num_blocks[4])
        self.conv_next = nn.ModuleList([self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5])
        self.num_features = num_features

    def forward(self, x):
        x = self.stem_conv(x)
        features = []
        for layer in self.conv_next:
            x = layer(x)
            features.append(x)
        return features[-self.num_features:]


def build_conv_next(stem_channels=32, feature_channels=None, num_features=3, num_blocks=None):
    if num_blocks is None:
        num_blocks = [3, 3, 3, 9, 3]
    if feature_channels is None:
        feature_channels = [64, 128, 256, 512, 1024]
    model = ConvNeXt(stem_channels, feature_channels, num_features, num_blocks)
    return model, feature_channels[-num_features:]
