import numpy as np
import pickle
import h5py
from scipy.misc import imread
import os

data_dir = os.path.expanduser('~/datasets/eecs442challenge/')

assert os.path.exists(data_dir)

def init():
    pass

def initialize(opt):
    return

def load_image(idx, is_train=True):
    if is_train:
        p = os.path.join(data_dir, 'train', 'color', str(idx) + '.png')
    else:
        p = os.path.join(data_dir, 'test', 'color', str(idx) + '.png')
    return imread(p,mode='RGB')

def load_mask(idx, is_train=True):
    if is_train:
        p = os.path.join(data_dir, 'train', 'mask', str(idx) + '.png')
    else:
        p = os.path.join(data_dir, 'test', 'mask', str(idx) + '.png')
    return imread(p,mode='L')

def load_gt(idx):
    p = os.path.join(data_dir, 'train', 'normal', str(idx) + '.png')
    return imread(p,mode='RGB')

def setup_val_split(opt = None):
    train = range(0, 19000)
    #train = range(10)
    return train, train

def get_test_set():
    return range(2000)
