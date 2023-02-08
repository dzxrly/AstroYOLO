import torch.nn as nn

from model.layers import CBL


class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):
        super(PredictNet, self).__init__()
        self.predict_conv = nn.ModuleList(
            [
                nn.Sequential(
                    CBL(feature_channels[i] // 2, feature_channels[i], 3),
                    nn.Conv2d(feature_channels[i], target_channels, 1),
                )
                for i in range(len(feature_channels))
            ]
        )

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]
        return predicts
