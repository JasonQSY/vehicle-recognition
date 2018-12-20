import numpy as np
import pandas as pd
import random
from sklearn import tree

def load_csv(path, label_only=False):
    buffer = np.loadtxt(path,dtype=np.str,delimiter=',',usecols=(0,1),skiprows=1)
    img_name = buffer[:,0]
    img_label = buffer[:,1].astype(np.int16)
    if label_only == True:
        return img_label
    return img_name, img_label

def pick_train_idx():
    img_label = load_csv('E:/避免根目录/my_dataset/ROB535_data/trainval/labels.csv',True)
    #num = [133,1147,993]
    num = [int(133*1.5),int(1147*1.5),int(993*1.5)]
    train_0 = []
    train_1 = []
    train_2 = []
    idx = []
    for i in range(0,7573):
        if img_label[i] == 0:
            train_0.append(i)
        elif img_label[i] == 1:
            train_1.append(i)
        else:
            train_2.append(i)
    random.shuffle(train_0)
    random.shuffle(train_1)
    random.shuffle(train_2)
    
    for j in range(0,num[0]):
        idx.append(train_0[j])
    for j in range(0,num[1]):
        idx.append(train_1[j])
    for j in range(0,num[2]):
        idx.append(train_2[j])
    random.shuffle(idx)
    return idx
    
    

def main():
    ########## system parameters ##########
    base_dir = 'E:/避免根目录/my_dataset/ROB535_data/'
    
    # prediction results on training set of each sub-classfier
    resnet50_train_dir = 'ensemble_res_trainset.npy'
    inception_v3_train_dir = 'ensemble_iv3_trainset.npy'
    inception_resnet_v2_train_dir = 'ensemble_v4_trainset.npy'
    vgg19_train_dir = 'ensemble_vgg19_trainset.npy'
    
    inception_v3_5e5_train_dir = 'ensemble_iv3_trainset_5e5.npy'
    inception_v3_5e5_f2_train_dir = 'ensemble_iv3_trainset_5e5_f2.npy'
    
    # prediction results on test set of each sub-classifier
    resnet50_test_dir = 'ensemble_res_result.csv'
    inception_v3_test_dir = 'task1_iv3_result.csv'
    inception_resnet_v2_test_dir = 'task1_v4_result.csv'
    vgg19_test_dir = 'task1_result.csv'
    
    inception_v3_5e5_test_dir = 'task1_iv3_result_5e5.csv'
    inception_v3_5e5_f2_test_dir = 'task1_iv3_result_5e5_f2.csv'
    
    ########## get sub-classifier prediction of each training image ##########
    sub_trainset_1 = np.load(base_dir + resnet50_train_dir)
    sub_trainset_2 = np.load(base_dir + inception_v3_train_dir)
    sub_trainset_3 = np.load(base_dir + inception_v3_5e5_train_dir)
    sub_trainset_4 = np.load(base_dir + inception_v3_5e5_f2_train_dir)
    sub_trainset_5 =  np.load(base_dir + vgg19_train_dir)
    
    trainset = np.stack((#sub_trainset_1,
                         sub_trainset_2,
                         sub_trainset_3),
                         #sub_trainset_4,
                         #sub_trainset_5),
                         axis=-1)
    
    ########## get sub-classifer prediction of each testing image ##########
    sub_testset_1 = load_csv(base_dir + resnet50_test_dir, True)
    sub_testset_2 = load_csv(base_dir + inception_v3_test_dir, True)
    sub_testset_3 = load_csv(base_dir + inception_v3_5e5_test_dir, True)
    test_img_name, sub_testset_4 = load_csv(base_dir + inception_v3_5e5_f2_test_dir, False)
    sub_testset_5 = load_csv(base_dir + vgg19_test_dir, True)
    
    testset = np.stack((#sub_testset_1,
                        sub_testset_2,
                        sub_testset_3),
                        #sub_testset_4,
                        #sub_testset_5,
                        axis=-1)
    
    ########## train a decision-tree classifier ##########
    idx = pick_train_idx()
    train_label = load_csv(base_dir + 'trainval/labels.csv', True)
    final_model = tree.DecisionTreeClassifier()
    final_model = final_model.fit(trainset[idx], train_label[idx])
    #final_model = final_model.fit(trainset, train_label)
    
    ########## predict on the test set and save outputs file ########## 
    predictions = final_model.predict(testset)
    
    result = []
    distribution = [0,0,0]
    for i in range(0,2631):
        img_name = test_img_name[i]
        y = predictions[i]
        result.append((img_name,y))
        distribution[y] = distribution[y] + 1
        print('image idx: %d, name: %s, prediciton result:%d'%(i,img_name,y))
    
    print('Finsh predictions')
    print('Write results to E:/避免根目录/my_dataset/ROB535_data/task1_tree_result.csv')
    out = pd.DataFrame(columns=['guid/image','label'],data=result)
    out.to_csv('E:/避免根目录/my_dataset/ROB535_data/task1_tree_result.csv',encoding='utf-8',index=False)    
    print('overall distribution')
    print('class 0: ', distribution[0]/2631.0)
    print('class 1: ', distribution[1]/2631.0)
    print('class 2: ', distribution[2]/2631.0)

    return
    
    
    

if __name__ == '__main__':
    main()

