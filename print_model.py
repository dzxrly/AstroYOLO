import torch
from torchinfo import summary

import config.model_config as cfg
from model.model import BuildModel

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BuildModel(use_trans=True).to(device)
    summary(model, input_size=[(1, 3, cfg.TRAIN['TRAIN_IMG_SIZE'], cfg.TRAIN['TRAIN_IMG_SIZE'])])
