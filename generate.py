import cv2
import torch
import tqdm
import os
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imsave, imresize
from glob import glob

import data.rob535_task1.ref as ds
import train as net


def preprocess(img):
    height, width = img.shape[0:2]
    if height >= width:
        width = int(224 / height * width)
        height = 224
    else:
        height = int(224 / width * height)
        width = 224
    img = cv2.resize(img, dsize=(width, height))
    #imsave('./save/origin_{}.png'.format(idx), img)
    img = (img / 255 - 0.5) * 2
    img_old = img
    img = np.zeros((224, 224, 3))
    img[round(112-height/2):round(112+height//2), round(112-width/2):round(112+width//2)] = img_old
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img = img[None, :, :, :]
    img = torch.FloatTensor(img)
    return img


def generate():
    """
    Generate label on the final test set
    """
    func, config = net.init()
    files = glob(ds.data_dir + 'test/*/*_image.jpg')
    print("find {} test images".format(len(files)))
    #results = {}
    f = open('result.csv', 'w')
    f.write('guid/image,label\n')

    for filename in files:
        img = imread(filename)
        img = preprocess(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = output['preds'][0]
        pred = np.argmax(pred)

        dirs = filename.split('/')
        fdir = dirs[-2]
        fname = dirs[-1][0:4]
        fname = fdir + '/' + fname
        #results[fname] = pred
        line = fname + ',' + str(pred)
        f.write(line + '\n')
        print(line)

    f.close()


def evaluate(test_set):
    func, config = net.init()
    num_correct = 0
    ds.init()

    for idx in tqdm(test_set):
        img = ds.load_image(idx, is_train)
        img = preprocess(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = output['preds'][0]
        pred = np.argmax(pred) # predicted label

        gt = ds.load_gt(idx)
        if pred == gt:
            num_correct += 1

    print("[summary]")
    print("validate {} images".format(len(test_set)))
    accu = num_correct / len(test_set)
    print("accuracy: {}".format(accu))


if __name__=='__main__':
    #test_set = range(0,1000)
    #generate(test_set, True)
    generate()
