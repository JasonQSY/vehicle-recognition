import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from scipy.misc import imresize, imsave

from utils.misc import get_transform, kpt_affine


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds, index):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']

        self.ds = ds
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(self.index[idx % len(self.index)])

    def loadImage(self, idx):
        ds = self.ds
        # load image
        img = ds.load_image(idx)

        # data argumentation (closed by default)
        #trans_aug = np.random.randint(8) * 45
        trans_aug = 0
        trans = cv2.getRotationMatrix2D((64,64), trans_aug, 1)
        trans_normal = np.eye(3)
        trans_normal[:2, :2] = trans[:2, :2]
        #flip_aug = np.random.randint(3)
        flip_aug = 2

        # for img
        img = cv2.warpAffine(img.astype(np.uint8), trans, (self.input_res, self.input_res))
        if flip_aug != 2:
            img = cv2.flip(img, flip_aug)
        img = (img / 255 - 0.5) * 2

        # for mask
        mask = ds.load_mask(idx)
        mask = cv2.warpAffine(mask.astype(np.uint8), trans, (self.output_res, self.output_res))
        if flip_aug != 2:
            mask = cv2.flip(mask, flip_aug)
        mask = mask / 255
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        # ground true
        gt = ds.load_gt(idx)
        gt_bg = gt[mask < 0.5, :].mean(axis=0)
        gt = cv2.warpAffine(gt.astype(np.uint8), trans, (self.output_res, self.output_res))
        if flip_aug != 2:
            gt = cv2.flip(gt, flip_aug)
        gt[mask < 0.5] = gt_bg
        gt = (gt / 255 - 0.5) * 2

        # normalize surface normal
        gt = np.transpose(gt, (2, 0, 1))
        gt = gt / np.linalg.norm(gt, axis=0)
        gt = np.transpose(gt, (1, 2, 0))

        # rotate normal
        gt = np.transpose(gt, (2, 0, 1))
        gt = np.reshape(gt, (3, 128 * 128))
        gt = np.matmul(trans_normal, gt)
        gt = np.reshape(gt, (3, 128, 128))
        gt = np.transpose(gt, (1, 2, 0))

        # normalize surface normal again
        gt = np.transpose(gt, (2, 0, 1))
        gt = gt / np.linalg.norm(gt, axis=0)
        gt = np.transpose(gt, (1, 2, 0))

        # flip normal
        if flip_aug == 1:
            gt[:, :, 0] = - gt[:, :, 0]
        elif flip_aug == 0:
            gt[:, :, 1] = - gt[:, :, 1]
        else: # 2 = no flip
            pass

        return img.astype(np.float32), mask.astype(np.float32), gt.astype(np.float32)


def init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    import ref as ds
    ds.init()

    train, valid = ds.setup_val_split()
    dataset = { key: Dataset(config, ds, data) for key, data in zip( ['train', 'valid'], [train, valid] ) }

    use_data_loader = config['train']['use_data_loader']

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=False, num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, masks, gts = next(loader)
            yield {
                'imgs': imgs,
                'masks': masks,
                'gts': gts,
            }

    return lambda key: gen(key)

if __name__ == "__main__":
    cf = {
        'inference': {
            'nstack': 8,
            'inp_dim': 256,
            'oup_dim': 3,
            'num_parts': 3,
            'increase': 128,
            'keys': ['imgs']
        },
        'train': {
            'batchsize': 5,
            'input_res': 128,
            'output_res': 128,
            'train_iters': 100,
            'valid_iters': 0,
            'num_workers': 1,
            'use_data_loader': True,
        },
    }
    func = init(cf)('train')
    for data in func:
        for i in range(5):
            imsave("{}_color.png".format(i), data['imgs'][i])
            imsave("{}_mask.png".format(i), data['masks'][i])
            imsave("{}_normal.png".format(i), data['gts'][i])
        exit(0)
