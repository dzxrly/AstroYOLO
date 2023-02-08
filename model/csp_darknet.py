import torch.nn as nn

from model.layers import CSPLayer, DownSampleConv, CBM


class CSPDarknet53(nn.Module):
    def __init__(self, stem_channels=32, feature_channels=None, num_features=3):
        super(CSPDarknet53, self).__init__()
        if feature_channels is None:
            feature_channels = [64, 128, 256, 512, 1024]
        self.stem_conv = CBM(3, stem_channels, 3)
        self.stage_1 = nn.Sequential(
            DownSampleConv(stem_channels, feature_channels[0]),
            CSPLayer(feature_channels[0], feature_channels[0], 1)
        )
        self.stage_2 = nn.Sequential(
            DownSampleConv(feature_channels[0], feature_channels[1]),
            CSPLayer(feature_channels[1], feature_channels[1], 2)
        )
        self.stage_3 = nn.Sequential(
            DownSampleConv(feature_channels[1], feature_channels[2]),
            CSPLayer(feature_channels[2], feature_channels[2], 8)
        )
        self.stage_4 = nn.Sequential(
            DownSampleConv(feature_channels[2], feature_channels[3]),
            CSPLayer(feature_channels[3], feature_channels[3], 8)
        )
        self.stage_5 = nn.Sequential(
            DownSampleConv(feature_channels[3], feature_channels[4]),
            CSPLayer(feature_channels[4], feature_channels[4], 4)
        )
        self.csp_darknet = nn.ModuleList([self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5])
        self.num_features = num_features

    def forward(self, x):
        x = self.stem_conv(x)
        features = []
        for layer in self.csp_darknet:
            x = layer(x)
            features.append(x)
        return features[-self.num_features:]


def build_darknet(stem_channels=32, feature_channels=None, num_features=3):
    if feature_channels is None:
        feature_channels = [64, 128, 256, 512, 1024]
    model = CSPDarknet53(stem_channels, feature_channels, num_features)
    return model, feature_channels[-num_features:]
