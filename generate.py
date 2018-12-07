import cv2
import torch
import tqdm
import os
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imsave, imresize
from glob import glob

import data.rob535_task2.ref as ds
import train as net


def preprocess(img):
    size = 224
    height, width = img.shape[0:2]
    if height >= width:
        width = int(size / height * width)
        height = size
    else:
        height = int(size / width * height)
        width = size
    img = cv2.resize(img, dsize=(width, height))
    #imsave('./save/origin_{}.png'.format(idx), img)
    img = (img / 255 - 0.5) * 2
    img_old = img
    img = np.zeros((size, size, 3))
    img[round(size//2-height/2):round(size//2+height//2), round(size//2-width/2):round(size//2+width//2)] = img_old
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

def generate_task2():
    """
    Generate task 2 result
    """
    func, config = net.init()
    files = glob(ds.data_dir + 'test/*/*_image.jpg')
    f = open('task2.csv', 'w')
    f.write('guid/image/axis,value\n')

    for filename in files:
        img = imread(filename)
        img = preprocess(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = output['preds'][0][0]

        dirs = filename.split('/')
        fdir = dirs[-2]
        fname = dirs[-1][0:4]
        fname = fdir + '/' + fname
        axis = ['x','y','z']
        for i in range(3):
            line = fname + '/' + axis[i] + ',' + str(pred[i])
            print(line)
            f.write(line + '\n')

    f.close()


def evaluate(test_set):
    func, config = net.init()
    num_correct = 0
    ds.init()

    for idx in tqdm(test_set):
        img = ds.load_image(idx, True)
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


def evaluate_task2(test_set):
    func, config = net.init()
    ds.init()
    error_sum = 0.

    for idx in tqdm(test_set):
        #tqdm.write(str(idx))
        img = ds.load_image(idx, True)
        img = preprocess(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = output['preds'][0][0]
        #tqdm.write("pred: " + str(pred))
        gt = ds.load_gt(idx)
        #tqdm.write("gt: " + str(gt))
        error = np.sqrt(((pred - gt)**2).sum())
        #tqdm.write(str(error))
        error_sum += error

    print("[summary]")
    rms = error_sum / len(test_set)
    print("RMSE: {}".format(rms))


if __name__=='__main__':
    test_set = range(0,1500)
    valid_set = range(3000, 4000)
    #evaluate(test_set)
    #evaluate(valid_set)
    #generate()
    generate_task2()
    #evaluate_task2(test_set)
    #evaluate_task2(valid_set)
