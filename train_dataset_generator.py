import sys
from typing import Dict, List, Tuple

import numpy as np
import os
from fits_config import fits_config as config
from utils.fits_operator import fits_reproject, create_dir
from tqdm import tqdm
import cv2
import argparse
import warnings
from to_annotation import xml_save


def save_process_img(bg_img: np.ndarray, save_path: str, window_coord_left_top: List[Tuple], window_size: int,
                     bbox_coord: List, fits_name: str):
    bg_img = np.squeeze(bg_img[0, :, :])
    for (h, w) in window_coord_left_top:
        bg_img[h:h + window_size, w] = 0.5
        bg_img[h:h + window_size, w + window_size] = 0.5
        bg_img[h, w:w + window_size] = 0.5
        bg_img[h + window_size, w:w + window_size] = 0.5
    bg_img[bbox_coord[0]:bbox_coord[2], bbox_coord[1]] = 1.0
    bg_img[bbox_coord[0]:bbox_coord[2], bbox_coord[3]] = 1.0
    bg_img[bbox_coord[0], bbox_coord[1]:bbox_coord[3]] = 1.0
    bg_img[bbox_coord[2], bbox_coord[1]:bbox_coord[3]] = 1.0
    cv2.imwrite(os.path.join(save_path, '{}_process.png'.format(fits_name)), bg_img * 255.0,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])


def fits_2_npy(fits_path: str, bbox: list, fits_name: str, npy_save_path: str, img_save_path: str, label_save_path: str,
               output_size: int = 352, padding: int = 10, window_number: int = 5) -> List[Dict]:
    """
    Convert fits to npy
    :param npy_save_path:
    :param img_save_path:
    :param label_save_path:
    :param fits_name:
    :param window_number:
    :param fits_path:
    :param bbox:
    :param output_size:
    :param padding:
    """
    used_filter = config['used_filter']
    reproject_target_filter = config['reproject_target_filter']
    fits_path_arr = []
    target_fits_path = ''
    for filename in os.listdir(fits_path):
        if filename.endswith('.fits.bz2') and filename.split('-')[1].lower() in used_filter.replace(
                reproject_target_filter, '').lower():
            fits_path_arr.append(os.path.join(fits_path, filename))
        if filename.endswith('.fits.bz2') and filename.split('-')[1].lower() == reproject_target_filter.lower():
            target_fits_path = os.path.join(fits_path, filename)
    fits_ndarr = fits_reproject({
        'target_fits_path': target_fits_path,
        'fits_without_target_path': fits_path_arr
    }, a=config['fit_reproject_a'], n_samples=config['fit_reproject_n_samples'],
        contrast=config['fit_reproject_contrast'])
    c, h, w = fits_ndarr.shape
    box_x, box_y = abs(bbox[0] - bbox[2]), abs(bbox[3] - bbox[1])
    origin_bg_ndarr = np.zeros((c, h + 2 * output_size, w + 2 * output_size))
    origin_bg_ndarr[:, output_size:output_size + h, output_size:output_size + w] = fits_ndarr
    top_left_coord_range = output_size - max(box_x, box_y)
    warnings.simplefilter('ignore', DeprecationWarning)
    top_left_hs = np.random.random_integers(bbox[0] + output_size - top_left_coord_range, bbox[0] + output_size,
                                            window_number)
    top_left_ws = np.random.random_integers(bbox[1] + output_size - top_left_coord_range, bbox[1] + output_size,
                                            window_number)
    window_bbox_top_left_coord = []
    for coord_h, coord_w in zip(top_left_hs, top_left_ws):
        assert np.sum(np.isnan(origin_bg_ndarr[:, coord_h:coord_h + output_size,
                               coord_w:coord_w + output_size])) == 0, '[Error]: \033[1;31m [Error]: NaN in ndarr \033[0m'
        window_bbox_top_left_coord.append(
            {'npy_img': origin_bg_ndarr[:, coord_h:coord_h + output_size, coord_w:coord_w + output_size],
             'bbox': [str(bbox[0] + output_size - coord_h), str(bbox[1] + output_size - coord_w),
                      str(bbox[0] + output_size - coord_h + box_x), str(bbox[1] + output_size - coord_w + box_y)]})
    save_imgs = []
    for index, obj in enumerate(window_bbox_top_left_coord):
        # Save Data
        np.save(os.path.join(npy_save_path, '{}_{}.npy'.format(fits_name, index)), obj['npy_img'])
        # Save Annotation
        xml_save(
            save_path=label_save_path,
            folder='VOC2007',
            filename='{}_{}.npy'.format(fits_name, index),
            width=output_size,
            height=output_size,
            depth=3,
            obj_name='bhb',
            left_up=[obj['bbox'][0], obj['bbox'][1]],
            right_down=[obj['bbox'][2], obj['bbox'][3]]
        )
        save_imgs.append({
            'saved_path': '{}_{}.npy'.format(fits_name, index),
            'bbox_class': '{},{}'.format(','.join(obj['bbox']), '0')
        })
    save_process_img(origin_bg_ndarr, img_save_path, [(h, w) for h, w in zip(top_left_hs, top_left_ws)], output_size,
                     [bbox[0] + output_size, bbox[1] + output_size, bbox[2] + output_size, bbox[3] + output_size],
                     fits_name)
    return save_imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert fits to npy')
    parser.add_argument('-m', '--model', type=str, help='train or valid or test ?', required=True)
    parser.add_argument('-c', '--cross', type=str)
    args = parser.parse_args()
    model = args.model
    cross_index = args.cross
    assert model in ['train', 'valid', 'test'], 'model must be train or valid or test'
    obj_txt_path = config['obj_txt_path']
    fits_parent_path = config['fits_parent_path']
    output_size = config['window_size']
    window_number = config['window_number']
    JPEG_save_path = config['JPEG_save_path'] if cross_index is None or cross_index == '' else config[
        'JPEG_save_path'].replace('bhb_dataset', 'bhb_dataset_{}'.format(cross_index))
    txt_save_path = config['txt_save_path'] if cross_index is None or cross_index == '' else config[
        'txt_save_path'].replace('bhb_dataset', 'bhb_dataset_{}'.format(cross_index))
    annotation_save_path = config['annotation_save_path'] if cross_index is None or cross_index == '' else \
        config[
            'annotation_save_path'].replace('bhb_dataset', 'bhb_dataset_{}'.format(cross_index))
    img_save_path = config['img_save_path'] if cross_index is None or cross_index == '' else config[
        'img_save_path'].replace('bhb_dataset', 'bhb_dataset_{}'.format(cross_index))
    create_dir(os.path.join(JPEG_save_path))
    create_dir(os.path.join(txt_save_path))
    create_dir(os.path.join(annotation_save_path))
    create_dir(os.path.join(img_save_path))
    train_txt = []
    train_dataset_count = len(open(os.path.join(obj_txt_path, '{}.txt'.format(
        model) if cross_index is None or cross_index == '' else '{}_{}.txt'.format(model, cross_index)),
                                   'r').readlines())
    with open(os.path.join(obj_txt_path,
                           '{}.txt'.format(
                               model) if cross_index is None or cross_index == '' else '{}_{}.txt'.format(
                               model, cross_index)), 'r') as f:
        with tqdm(total=train_dataset_count, ncols=100,
                  desc='Generating {} data'.format(model)) as pbar:
            for line in f:
                line = line.strip()
                fits_path, bbox_class = line.split(' ')
                fits_path = fits_path.split('/')[-1]
                fits_name = fits_path
                pbar.set_description('Generating {} data: {}'.format(model, fits_path))
                fits_path = os.path.join(fits_parent_path, fits_path)
                bbox = [int(bbox_class.split(',')[i]) for i in range(4)]
                saved = fits_2_npy(
                    fits_path=fits_path,
                    bbox=bbox,
                    fits_name=fits_name,
                    npy_save_path=JPEG_save_path,
                    img_save_path=img_save_path,
                    label_save_path=annotation_save_path,
                    output_size=output_size,
                    window_number=window_number if model == 'train' else 1
                )
                for obj in saved:
                    train_txt.append(obj['saved_path'].split('.npy')[0] + ' -1')
                pbar.update(1)
            f.close()

    with open(os.path.join(txt_save_path, 'bhb_{}.txt'.format(model)), 'w') as f:
        for line in train_txt:
            f.write(line + '\n')
        f.close()
