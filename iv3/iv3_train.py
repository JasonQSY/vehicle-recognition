# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:29:12 2018

@author: Team31_ROB535_18fall
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from PIL import Image


def load_csv(path):
    buffer = np.loadtxt(path,dtype=np.str,delimiter=',',usecols=(0,1),skiprows=1)
    #print(buffer.shape)
    img_name = buffer[:,0]
    img_label = buffer[:,1].astype(np.int16)
    return img_name, img_label

def generate_imgarray_from_file(img_names, labels, 
                height, width, batch_size, start, base_dir):
    '''This generator function creates a generator object, which contains
    batch_size of tuples. Each tuple is a training image with its label. We
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

def cnn_input_preprocess(x):
    ''' Take a 3D numpy array as input and return a 3D numpy array, called by ImageDataGenerator.
       Preprocessing of Inception_V3 is (x/255 - 0.5)*2
    '''
    x = (x/255.0 - 0.5) * 2
    return x
        
def train_model(base_dir):
    train_dir = 'E:/避免根目录/my_dataset/ROB535_data/train'
    val_dir = 'E:/避免根目录/my_dataset/ROB535_data/val'
    start_from_checkpoint = True
    checkpoint_path = 'E:/避免根目录/my_dataset/ROB535_data/task1_iv3_model.h5'     

    ########## model parameters ##########
    input_height, input_width = 299, 299
    our_batch_size = 32
    our_epochs = 30
    num_layer_freeze = 211   # MAX = 311
    num_train_batches = 165
    num_val_batches = 70
    
    ########## model architecture ##########
    if start_from_checkpoint == True:
       our_model = models.load_model(checkpoint_path)
       print('Start from checkpoint: ',checkpoint_path)
    else:
       print('Start from the beginning')
       # load a pretrained Inception_V3 network and remove its output layers 
       base_model = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_tensor=None,
                  input_shape=(input_height, input_width, 3))
       print('total number of layers in Inception-V3: ', len(base_model.layers))
       # freeze the layers we don't want to train
       for layer in base_model.layers[:num_layer_freeze]:
           layer.trainable = False
       # add our own layers
       base_out = base_model.output   # shape: num_imgs*8*8*1536
       x = layers.AveragePooling2D(pool_size=(8,8), strides=(8,8))(base_out)
       x = layers.Flatten()(x)
       predictions = layers.Dense(3, activation='softmax')(x)
       # set the inputs and outputs and compile the model
       our_model = models.Model(inputs=base_model.input, outputs=predictions)
       our_optimizer = optimizers.SGD(
            lr=0.0001,momentum = 0.9,decay = 0.0,nesterov = True)
       our_model.compile(
            loss='categorical_crossentropy',
            optimizer = our_optimizer, metrics = ['accuracy'])
    
    print(our_model.summary())
    
    ########## data pipeline ##########
    
    # create a training data generator 
    
    train_datagen = ImageDataGenerator(
            rescale = None,
            zoom_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False,
            width_shift_range = 0.1,
            preprocessing_function = cnn_input_preprocess)
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = (input_height, input_width),
            batch_size = our_batch_size,
            shuffle = True,
            class_mode = 'categorical')
            
    # create a validation data generator
    val_datagen = ImageDataGenerator(
          rescale = None,
          preprocessing_function = cnn_input_preprocess)
    val_generator = val_datagen.flow_from_directory(
          val_dir,
          target_size = (input_height, input_width),
          batch_size = our_batch_size,
          shuffle = False,
          class_mode = 'categorical')

    
    ########## checkpoint and model save ##########
    checkpoint = ModelCheckpoint(
            'task1_iv3_model.h5',monitor='val_acc',verbose=1,
            save_best_only=True,save_weights_only=False,mode='auto',period=1)
    early_stop = EarlyStopping(monitor='val_acc',
                            min_delta=0,patience=5,verbose=1,mode='auto')
    
    ########## model training ##########
    our_model.fit_generator(
        train_generator,
        steps_per_epoch = num_train_batches,
        epochs = our_epochs,
        validation_data = val_generator,    
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
