import numpy as np
import os
import glob as gb
import pandas as pd
from keras import models
from keras.preprocessing import image

''' test pretrained inception_v3'''

def cnn_input_preprocess(x):
    ''' take a numpy array as input and return a numpy array'''
    x = (x/255.0 - 0.5) * 2
    return x

def test():
    model_path = 'E:/避免根目录/my_dataset/ROB535_data/task1_iv3_model.h5'
    test_base_path = 'E:/避免根目录/my_dataset/ROB535_data/test/'
    L = len(test_base_path)
    height, width = 299, 299
    folder_size = [53,65,95,151,51,54,
                   195,168,68,101,221,58,
                   58,183,54,280,100,193,
                   68,86,109,143,77]   # number of images inside each folder
    num_aug = 5   # number of predictions for each test image, the final result is based on voting

    print('Loading the trained model from: ', model_path)
    final_model = models.load_model(model_path)
    
    test_datagen = image.ImageDataGenerator(
               rescale = None,
               zoom_range = 0.1,
               width_shift_range = 0.1,
               horizontal_flip = True,
               preprocessing_function = cnn_input_preprocess)

    print('Getting test image folders')
    folders = os.listdir(test_base_path)
    folders.sort()
    N = len(folders)
    tail = 1
    result = []
    distribution = [0,0,0]
    
    for i in range(0,N):
        print('Passing folder %d'%(i+1))
        img_paths = gb.glob(test_base_path + folders[i] + '/*.jpg')
        num = len(img_paths)
        l = len(os.path.dirname(img_paths[0]))
        idx = []
        for k in range(0,num):
            index = img_paths[k][l+1:-10]
            idx.append(index)
        idx.sort()
        for j in range(0,num):
            img = image.load_img(img_paths[j], target_size=(height, width))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            votes = np.array([0,0,0])   # number of votes of class 0,1,2
            for k in range(0,num_aug):
                y = final_model.predict_generator(test_datagen.flow(x,batch_size=1), steps=1)   # get probabilities of each class
                z = np.argmax(y, axis=1)   # pick the class label corresponding to the largest probability
                votes[z] = votes[z] + 1
            prediction = np.argmax(votes)   # pick the label appearing for most times
            img_name = folders[i] + '/' + idx[j]
            result.append((img_name, prediction))
            print('image idx: %d, name: %s, freq_0:%d, freq_1:%d, freq_2:%d, prediciton result:%d'%(tail,img_name,votes[0],votes[1],votes[2],prediction))
            tail = tail+1
            distribution[prediction] = distribution[prediction] + 1
    
        
    print('Finsh predictions')
    print('Write results to csv')
    out = pd.DataFrame(columns=['guid/image','label'],data=result)
    out.to_csv('E:/避免根目录/my_dataset/ROB535_data/task1_iv3_result.csv',encoding='utf-8',index=False)    
    print('overall_distribution: ')
    print('class 0: ', distribution[0]/2631.0)
    print('class 1: ', distribution[1]/2631.0)
    print('class 2: ', distribution[2]/2631.0)

    return
 

if __name__ == '__main__':
    test()

