import cv2
import torch
import tqdm
import os
import numpy as np
from tqdm import tqdm
from scipy.misc import imsave, imresize

import data.eecs442_challenge.ref as ds
import train as net

def generate(test_set, is_train=True):
    func, config = net.init()

    for idx in tqdm(test_set):
        img = ds.load_image(idx, is_train)
        #imsave('./save/origin_{}.png'.format(idx), img)
        img = (img / 255 - 0.5) * 2
        img = img.astype(np.float32)
        img = img[None, :, :, :]
        img = torch.FloatTensor(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = torch.FloatTensor(output['preds'][0][:, -1])
        pred = pred[0, :, :, :]
        pred = pred.permute(1, 2, 0)
        for i in range(3):
            pred[:,:,i] = pred[:,:,i] / torch.norm(pred, 2, 2)

        pred = (pred / 2 + 0.5) * 255
        pred = pred.numpy()
        pred = pred.astype(np.uint8)
        #pred = imresize(pred, (128, 128))
        imsave('./save/{}.png'.format(idx), pred)


if __name__=='__main__':
    test_set = range(0,1000)
    generate(test_set, True)
