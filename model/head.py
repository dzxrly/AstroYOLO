import torch
import torch.nn as nn


class YOLOHead(nn.Module):
    def __init__(self, num_classes, anchors, stride):
        super(YOLOHead, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, p):
        batch_size, n_g = p.shape[0], p.shape[-1]
        p = p.view(batch_size, self.num_anchors, 5 + self.num_classes, n_g, n_g).permute(0, 3, 4, 1, 2)
        p_de = self.__decode(p.clone())
        return p, p_de

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.stride
        anchors = (1.0 * self.anchors).to(device)
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = (
            grid_xy.unsqueeze(0)
            .unsqueeze(3)
            .repeat(batch_size, 1, 1, 3, 1)
            .float()
            .to(device)
        )
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        return pred_bbox.view(-1, 5 + self.num_classes) if not self.training else pred_bbox
