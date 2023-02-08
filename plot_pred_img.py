import math
import os
import shutil
import xml.etree.ElementTree as ET
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from utils.fits_operator import create_dir, save_bbox_img
from utils.utils import bboxes_iou


def read_bbox_from_xml(xml_path: str) -> List[int]:
    xml_tree = ET.parse(xml_path)
    bbox = xml_tree.find('object').find('bndbox')
    return [int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)]


if __name__ == '__main__':
    pred_txt_path = os.path.join('./pred_result', 'comp4_det_test_bhb.txt')
    label_txt_path = os.path.join('./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
    annotation_file_path = os.path.join('./data/VOCdevkit/VOC2007/Annotations')
    fits_file_path = os.path.join('./data/VOCdevkit/VOC2007/JPEGImages')
    img_save_path = os.path.join('./pred_img')
    img_save_path_with_low_iou = os.path.join('./pred_img_with_low_iou')
    iou_threshold = 0.2

    txt_content = {}
    pred_res = {}
    label_res = {}

    with open(pred_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            assert len(line) == 6, 'line length is not 6'
            img_name = line[0]
            cof = float(line[1])
            bbox = [int(line[i]) for i in range(2, 6)]
            if pred_res.__contains__(img_name):
                if pred_res[img_name]['cof'] < cof:
                    pred_res[img_name] = {'cof': cof, 'bbox': bbox}
            else:
                pred_res[img_name] = {'cof': cof, 'bbox': bbox}

    with open(label_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            img_id = line.split(' ')[0]
            label = read_bbox_from_xml(os.path.join(annotation_file_path, img_id + '.xml'))
            label_res[img_id] = {
                'label': label
            }

    create_dir(img_save_path)
    create_dir(img_save_path_with_low_iou)
    pred_iou = 0.0
    with tqdm(total=len(label_res), unit='img', desc='Plotting...', ncols=100) as pbar:
        for img_id in pred_res:
            if label_res.__contains__(img_id):
                pbar.set_description('Plotting {}'.format(img_id))
                pred_bbox = pred_res[img_id]['bbox']
                pred_score = pred_res[img_id]['cof']
                label_bbox = label_res[img_id]['label']
                fits_img = np.load(os.path.join(fits_file_path, img_id + '.npy'))
                save_bbox_img(img_save_path, img_id, fits_img, np.asarray(pred_bbox), np.asarray(pred_score),
                              np.asarray(label_bbox))
                pred_bbox_tensor = torch.as_tensor(np.expand_dims(np.asarray(pred_bbox), axis=0)).float()
                label_bbox_tensor = torch.as_tensor(np.expand_dims(np.asarray(label_bbox), axis=0)).float()
                iou = float(bboxes_iou(pred_bbox_tensor, label_bbox_tensor)[0][0].numpy())
                if iou < iou_threshold:
                    shutil.copy(
                        os.path.join(img_save_path, '{}_score={}.png'.format(img_id, math.floor(pred_score * 100))),
                        img_save_path_with_low_iou)
                pbar.set_postfix({
                    'iou': '{:.4f}'.format(iou)
                })
                pred_iou += iou
                pbar.update(1)
            else:
                print('Waring: {} is not in label_res'.format(img_id))
    print('pred_iou: {:.4f}'.format(pred_iou / len(label_res)))
