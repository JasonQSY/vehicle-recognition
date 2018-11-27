# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:29:12 2018

@author: Team31_ROB535_18fall
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from PIL import Image


def tmp_draft():   # just for temporal testing
    base_dir = 'E:/避免根目录/my_dataset/ROB535_data/trainval'
    input_height, input_width = 480, 640
    img_name, img_label = load_csv('E:/避免根目录/my_dataset/ROB535_data/trainval/labels.csv')
    my_generator = generate_imgarray_from_file(
        img_name,img_label,input_height,input_width,5,0,base_dir)
    for data in my_generator:
        num = data[1].shape[0]
        for n in range(0,num):
            print('image shape: ', data[0][n][:][:][:].shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(data[0][n][:][:][:]/255)
            print('label is: ', data[1][n][:])
        
    return

def load_csv(path):
    buffer = np.loadtxt(path,dtype=np.str,delimiter=',',usecols=(0,1),skiprows=1)
    #print(buffer.shape)
    img_name = buffer[:,0]
    img_label = buffer[:,1].astype(np.int16)
    return img_name, img_label

def generate_imgarray_from_file(img_names, labels, 
                height, width, batch_size, start, base_dir):
    '''This generator function creates a generator object, which contains
    batch_size tuples. Each tuple is a training image with its label. We
    use this function for keras.model.fit_generator'''
    while True:
        n = labels.shape[0]   # number of samples of the dataset
        idx = random.sample(range(start,n),batch_size)
        feature_list = []
        label_list = []
        for i in idx:
            img_path = img_names[i] + '_image.jpg'
            img_label = to_categorical(labels[i], num_classes=3)   # one-hot encode
            img = Image.open(base_dir + '/' + img_path)
            img = img.resize((width,height), resample=1)
            img_array = np.array(img)
            feature_list.append(img_array)
            label_list.append(img_label)
        batch_labels = np.stack(label_list, axis=0)   
        batch_features = np.stack(feature_list, axis=0)
        yield (batch_features, batch_labels)
        
def train_model(base_dir):
    ########## model parameters ##########
    input_height, input_width = 299, 299
    our_batch_size = 2
    our_epochs = 10
    num_layer_freeze = 15
    train_start_idx = 0
    val_start_idx = 6050
    #num_train_batches = 90
    #num_val_batches = 20
    num_train_batches = 5
    num_val_batches = 2
    
    ########## model architecture ##########
    # load a pretrained InceptionV3 network and remove the output layers 
    base_model = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(input_height, input_width, 3))
    # freeze the layers we don't want to train
    for layer in base_model.layers[:num_layer_freeze]:
        layer.trainable = False
        
    # add our own layers
    base_out = base_model.output   # shape: num_imgs*8*8*2048
    x = layers.Flatten()(base_out)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(3, activation='softmax')(x)
    
    our_model = models.Model(inputs=base_model.input, outputs=predictions)
    our_optimizer = optimizers.SGD(
            lr=0.0001,momentum = 0.9,decay = 0.0,nesterov = True)
    our_model.compile(
            loss='categorical_crossentropy',
            optimizer = our_optimizer, metrics = ['accuracy'])
    
    ########## data pipeline ##########
    img_name, img_label = load_csv(base_dir + '/labels.csv')
    ''' 7573 training images total, we split 20% as validation images'''
    '''
    # create a training data generator 
    train_datagen = generate_imgarray_from_file(img_name, img_label, 
            input_height, input_width, our_batch_size, 
            train_start_idx, base_dir)
    # create a validation data generator
    val_datagen = generate_imgarray_from_file(img_name, img_label, 
            input_height, input_width, our_batch_size, 
            val_start_idx, base_dir)
    '''
    '''
    for data in val_datagen:
        num = data[1].shape[0]
        for n in range(0,num):
            print('image shape: ', data[0][n][:][:][:].shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(data[0][n][:][:][:]/255)
            print('label is: ', data[1][n][:])
    '''
    
    ########## checkpoint and model save ##########
    checkpoint = ModelCheckpoint(
            'our_model.h5',monitor='val_acc',verbose=1,
            save_best_only=True,save_weights_only=False,mode='auto',period=1)
    early_stop = EarlyStopping(monitor='val_acc',
                            min_delta=0,patience=10,verbose=1,mode='auto')
    
    ########## model training ##########
    our_model.fit_generator(
        generate_imgarray_from_file(img_name, img_label, 
            input_height, input_width, our_batch_size, 
            train_start_idx, base_dir),   # training data generator
        steps_per_epoch = num_train_batches,
        epochs = our_epochs,
        validation_data = generate_imgarray_from_file(img_name, img_label, 
            input_height, input_width, our_batch_size, 
            val_start_idx, base_dir),    # validation data generator
        validation_steps = num_val_batches,
        callbacks = [checkpoint, early_stop],
        initial_epoch=0)
    
    
    return


def main():
    ########## system parameters ##########
    base_dir = 'E:/避免根目录/my_dataset/ROB535_data/trainval'
    
    # tmp_draft()
    train_model(base_dir)
   
    

if __name__ == '__main__':
    main()
