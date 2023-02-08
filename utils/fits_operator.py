import math
import os
import warnings
from typing import Dict

import cv2
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import MinMaxInterval, SqrtStretch
from reproject import reproject_interp


def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def fits_reproject(fits_path_obj: Dict, a: int = 70, n_samples: int = 500, contrast: float = 0.25) -> np.ndarray:
    """
    Reproject a FITS image to a target header
    :param fits_path_obj: Dict
    :param a: int
    :param n_samples: int
    :param contrast: float
    :return: C x H x W numpy array
    """
    # Target
    warnings.simplefilter('ignore', AstropyWarning)
    target_hdu = fits.open(fits_path_obj['target_fits_path'])[0]
    target_header = target_hdu.header
    target_data = target_hdu.data
    target_data = SqrtStretch()(MinMaxInterval()(target_data, clip=False))
    stack_img = np.expand_dims(target_data, axis=-1)
    # Other
    for fits_path in fits_path_obj['fits_without_target_path']:
        warnings.simplefilter('ignore', AstropyWarning)
        hdu = fits.open(fits_path)[0]
        reprojected_data, reprojected_footprint = reproject_interp(hdu, target_header)
        reprojected_data = SqrtStretch()(MinMaxInterval()(reprojected_data, clip=False))
        stack_img = np.concatenate((stack_img, np.expand_dims(reprojected_data, axis=-1)), axis=-1)
    # NaN to 0
    stack_img = np.where(np.isnan(stack_img), 0, stack_img)
    return np.asarray(stack_img.transpose((2, 1, 0)), dtype=np.float32)


def save_nan_error_img(fits_ndarr: np.ndarray) -> None:
    print(np.min(fits_ndarr), np.max(fits_ndarr))
    print(np.sum(np.isnan(fits_ndarr)))
    nan_map = np.expand_dims(np.cumsum(np.isnan(fits_ndarr), axis=0)[-1], axis=0)
    fits_ndarr = fits_ndarr + np.repeat(nan_map, fits_ndarr.shape[0], axis=0)
    cv2.imwrite('./nan_map.png', np.transpose(fits_ndarr[:3, :, :], (1, 2, 0)) * 255)


def save_bbox_img(save_path: str, fits_name: str, fits_ndarr: np.ndarray, pred_bbox: np.ndarray, pred_score: np.ndarray,
                  gt_bbox: np.ndarray) -> None:
    c, h, w = fits_ndarr.shape
    assert c >= 3, '[Error]: channel number must be >= 3'
    fits_ndarr = fits_ndarr[:3, :, :] * 255 if c > 3 else fits_ndarr * 255
    pred_bbox = np.where(pred_bbox < 0, 0, pred_bbox)
    pred_bbox = np.where(pred_bbox >= w, w - 1, pred_bbox)
    gt_bbox = np.where(gt_bbox < 0, 0, gt_bbox)
    gt_bbox = np.where(gt_bbox >= w, w - 1, gt_bbox)
    # Plot pred bbox
    fits_ndarr[0, int(pred_bbox[0]):int(pred_bbox[2]), int(pred_bbox[1])] = 255
    fits_ndarr[0, int(pred_bbox[0]):int(pred_bbox[2]), int(pred_bbox[3])] = 255
    fits_ndarr[0, int(pred_bbox[0]), int(pred_bbox[1]):int(pred_bbox[3])] = 255
    fits_ndarr[0, int(pred_bbox[2]), int(pred_bbox[1]):int(pred_bbox[3])] = 255
    # Plot gt bbox
    fits_ndarr[1, int(gt_bbox[0]):int(gt_bbox[2]), int(gt_bbox[1])] = 255
    fits_ndarr[1, int(gt_bbox[0]):int(gt_bbox[2]), int(gt_bbox[3])] = 255
    fits_ndarr[1, int(gt_bbox[0]), int(gt_bbox[1]):int(gt_bbox[3])] = 255
    fits_ndarr[1, int(gt_bbox[2]), int(gt_bbox[1]):int(gt_bbox[3])] = 255

    try:
        cv2.imwrite(os.path.join(save_path, '{}_score={}.png'.format(fits_name, math.floor(pred_score * 100))),
                    np.transpose(fits_ndarr, (1, 2, 0)))
    except Exception as e:
        print('[Warning]: save pred img failed: {}'.format(e))
        pass
