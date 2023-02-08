import sys

import torch
import torch.nn as nn

from model.conv_next import build_conv_next
from model.csp_darknet import build_darknet
from model.head import YOLOHead
from model.layers import SpatialPyramidPooling
from model.panet_conv import PANet as PANetConv
from model.panet_trans import PANet as PANetTrans
from model.predict_net import PredictNet

sys.path.append('..')
import config.model_config as cfg


class YOLO(nn.Module):
    def __init__(self, out_channels=255, use_trans=True, backbone='darknet'):
        super(YOLO, self).__init__()
        assert backbone in ['darknet', 'convnext',
                            'resnet'], 'Error: backbone must be in ["darknet", "convnex", "resnet"]'
        if backbone == 'darknet':
            self.backbone, feature_channels = build_darknet()
        elif backbone == 'convnext':
            self.backbone, feature_channels = build_conv_next()
        self.spp = SpatialPyramidPooling(feature_channels)
        self.panet = PANetTrans(feature_channels) if use_trans else PANetConv(feature_channels)
        self.predict_net = PredictNet(feature_channels, out_channels)

    def forward(self, x):
        features = self.backbone(x)
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        predicts = self.predict_net(features)
        return predicts


class BuildModel(nn.Module):
    def __init__(self, use_trans=True):
        super(BuildModel, self).__init__()
        self.anchors = torch.FloatTensor(cfg.MODEL['ANCHORS'])
        self.strides = torch.FloatTensor(cfg.MODEL['STRIDES'])
        self.num_classes = cfg.Customer_DATA['NUM']
        self.out_channel = cfg.MODEL['ANCHORS_PER_SCLAE'] * (self.num_classes + 5)
        self.yolo_v4 = YOLO(self.out_channel, use_trans)
        # Small Head
        self.head_s = YOLOHead(self.num_classes, self.anchors[0], self.strides[0])
        # Medium Head
        self.head_m = YOLOHead(self.num_classes, self.anchors[1], self.strides[1])
        # Large Head
        self.head_l = YOLOHead(self.num_classes, self.anchors[2], self.strides[2])

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.yolo_v4(x)
        out.append(self.head_s(x_s))
        out.append(self.head_m(x_m))
        out.append(self.head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
