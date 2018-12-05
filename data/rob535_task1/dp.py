import sys
import os
import torch
import numpy as np
import torch.utils.data
from scipy.misc import imresize, imsave
import cv2


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
        size = 1024
        # load image
        img = ds.load_image(idx)
        height, width = img.shape[0:2]
        if height >= width:
            width = int(size / height * width)
            height = size
        else:
            height = int(size / width * height)
            width = size
        img = cv2.resize(img, dsize = (width, height))
        img = (img / 255 - 0.5) * 2
        img_old = img
        img = np.zeros((size, size, 3))
        img[round(size//2 - height/2):round(size//2+height//2), round(size//2-width/2):round(size//2+width//2)] = img_old
        #img = (img / 255 - 0.5) * 2
        img = np.transpose(img, (2, 0, 1))

        # ground true
        gt = ds.load_gt(idx)

        #return img.astype(np.float32), gt
        return {
            'imgs': img.astype(np.float32),
            'gts': gt,
        }


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
        if key == 'valid':
            continue
        print(len(dataset[key]))
        num_step = len(dataset[key]) // batchsize
        print(num_step)
        config[key]['num_step'] = num_step
        loaders[key] = torch.utils.data.DataLoader(dataset[key],
                                                   batch_size=batchsize,
                                                   shuffle=False,
                                                   num_workers=config['train']['num_workers'],
                                                   pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        #batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, gts = next(loader)
            yield {
                'imgs': imgs,
                #'masks': masks,
                'gts': gts,
            }

    return lambda phase: loaders[phase].__iter__()

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






    # for data in func:
    #     for i in range(5):
    #         imsave("{}_color.png".format(i), data['imgs'][i])
    #         imsave("{}_mask.png".format(i), data['masks'][i])
    #         imsave("{}_normal.png".format(i), data['gts'][i])
    #     exit(0)
