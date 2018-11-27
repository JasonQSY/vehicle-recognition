#!/usr/bin/env python
# Evaluate the mean angle error of predicted normals.
# Dawei Yang
# ydawei@umich.edu


from __future__ import print_function

import argparse
import os
import imageio
import numpy as np


def scan_png_files(folder):
    '''
    folder: 1.png 3.png 4.png 6.png 7.exr unknown.mpeg
    return: ['1.png', '3.png', '4.png']
    '''
    ext = '.png'
    ret = [fname for fname in os.listdir(folder) if fname.endswith(ext)]

    return ret


def evaluate(prediction_folder, groundtruth_folder, mask_folder):
    '''
    Evaluate mean angle error of predictions in the prediction folder,
    given the groundtruth and mask images.
    '''
    # Scan folders to obtain png files
    if mask_folder is None:
        mask_folder = os.path.join(groundtruth_folder, '..', 'mask')

    pred_pngs = scan_png_files(prediction_folder)
    gt_pngs = scan_png_files(groundtruth_folder)
    mask_pngs = scan_png_files(mask_folder)

    pred_diff_gt = set(pred_pngs).difference(gt_pngs)
    assert len(pred_diff_gt) == 0, \
        'No corresponding groundtruth file for the following files:\n' + '\n'.join(pred_diff_gt)
    pred_diff_mask = set(pred_pngs).difference(mask_pngs)
    assert len(pred_diff_mask) == 0, \
        'No corresponding mask file for the following files:\n' + '\n'.join(pred_diff_mask)

    # Measure: mean angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for fname in pred_pngs:
        print('Proccessing file {}'.format(fname))
        prediction = imageio.imread(os.path.join(prediction_folder, fname))
        groundtruth = imageio.imread(os.path.join(groundtruth_folder, fname))
        mask = imageio.imread(os.path.join(mask_folder, fname)) # Greyscale image

        prediction = ((prediction / 255.0) - 0.5) * 2
        groundtruth = ((groundtruth / 255.0) - 0.5) * 2

        total_pixels += np.count_nonzero(mask)
        mask = mask != 0

        a11 = np.sum(prediction * prediction, axis=2)[mask]
        a22 = np.sum(groundtruth * groundtruth, axis=2)[mask]
        a12 = np.sum(prediction * groundtruth, axis=2)[mask]

        cos_dist = a12 / np.sqrt(a11 * a22)
        cos_dist[np.isnan(cos_dist)] = -1
        cos_dist = np.clip(cos_dist, -1, 1)
        angle_error = np.arccos(cos_dist)
        mean_angle_error += np.sum(angle_error)

    return mean_angle_error / total_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_folder', '-p', required=True,
                        help='Folder path of normal predictions')
    parser.add_argument('--groundtruth_folder', '-g', required=True,
                        help='Folder path of groundtruths')
    parser.add_argument('--mask_folder', '-m',
                        help='Mask folder, default to {groundtruth_folder}/../mask')

    args = parser.parse_args()
    mae = evaluate(**vars(args))
    print('Mean angle error: {}'.format(mae))
