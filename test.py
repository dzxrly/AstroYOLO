import os

import numpy as np
import torch
from tqdm import tqdm

import config.model_config as cfg
import utils.gpu as gpu
from eval.evaluator import Evaluator
from model.model import BuildModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def pred(pred_model, pred_class_num, iou_thresh=0.5):
    mAP = 0.0
    with torch.no_grad():
        APs, inference_time = Evaluator(pred_model, dataset_type='test').APs_voc(iou_thresh=iou_thresh)
        for i in APs:
            mAP += APs[i]
        mAP /= pred_class_num
    return mAP, inference_time


if __name__ == '__main__':
    model_path = os.path.join('./files/your model path/best.pt')
    print('Loading model from [{}]'.format(model_path))
    device = gpu.select_device(id=0)
    model = BuildModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    classes = cfg.Customer_DATA['CLASSES']
    class_num = len(classes) - 1 if 'unknown' in classes else len(classes)
    print('=' * 50)
    for i in range(len(classes)):
        print('Find Class {}: {}'.format(i + 1, classes[i]), '[Ignore]' if classes[i] == 'unknown' else '')
    print('Total Class Num: {}'.format(class_num))
    print('=' * 50)
    print('Test Starting...')
    mAP = []
    for threshold in tqdm(np.arange(0.5, 1.0, 0.05)):
        print('=' * 50)
        print('Threshold@{} Test'.format(int(threshold * 100)))
        ap, inference_time = pred(model, class_num, iou_thresh=threshold)
        mAP.append(ap)
        print('mAP: {}'.format(ap))
        print('inference_time: {}'.format(inference_time))
        print('=' * 50)
    print('All APs: {}'.format(mAP))
    print('=' * 50)
    for map_type in [50, 75, 95]:
        if map_type != 50:
            print('Select AP Index: {}'.format([i for i in range(int((map_type - 50) / 5) + 1)]))
            ap_total = sum(mAP[:int((map_type - 50) / 5) + 1])
            print(
                'mAP@{}: {:.2f}% - {}'.format(map_type, ap_total / (int((map_type - 50) / 5) + 1) * 100,
                                              ap_total / (int((map_type - 50) / 5) + 1)))
        else:
            print('mAP@{}: {:.2f}% - {}'.format(map_type, mAP[0] * 100, mAP[0]))
    print('=' * 50)
