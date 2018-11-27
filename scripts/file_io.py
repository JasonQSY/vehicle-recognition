import os
import numpy as np
import csv
from itertools import izip


def file_read(home_dir):
	"""
	read the image files name with there corresponding folder name
	the names are sorted from lower number to higeher alphabetic letter
	INPUT:
		home_dir: the directory above the folder 'test' and 'trainval'
	OUTPUT:
		train_image_list: the trainning image name list, the image name used to have '_image.jpg' after the name
		test_image_list: the testing image name list, the image name used to have '_image.jpg' after the name
	"""
	top_dir_test = home_dir + 'test/'
	top_dir_train = home_dir + 'trainval/'

	''' The test data  '''
	# read the folder name
	for root, test_dirs, files in os.walk(top_dir_test, topdown=False):
		for name in test_dirs:
			pass
	test_dirs.sort() # sort the directory with respect to the first letter
	# read the image file name
	test_image_list = []
	for i in range(np.shape(test_dirs)[0]):
		for root, dirs, files in os.walk(top_dir_test + test_dirs[i], topdown=False):
			for file in files:
				if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
					test_image_list.append(test_dirs[i]+'/'+file.split('_')[0])
	test_image_list.sort()

	''' The Train data  '''
	# read the folder name
	for root, train_dirs, files in os.walk(top_dir_train, topdown=False):
		for name in train_dirs:
			pass
	train_dirs.sort() # sort the directory with respect to the first letter
	# read the image file name
	train_image_list = []
	for i in range(np.shape(train_dirs)[0]):
		for root, dirs, files in os.walk(top_dir_train + train_dirs[i], topdown=False):
			for file in files:
				if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
					train_image_list.append(train_dirs[i]+'/'+file.split('_')[0])
	train_image_list.sort()

	return train_image_list, test_image_list



def classes_output(classification, test_image_list):
	"""
	combine the classification result and output the csv file available for upload
	INPUT:
		classification: the result of the calification
		test_image_list: the testing image name list
	OUTPUT:
		a file named 'classes.csv' is generated
	"""
	with open("classes.txt", "w") as text_file:
		text_file.write('guid/image,label\n')
		for i in range(np.shape(test_image_list)[0]):
			text_file.write(test_image_list[i] + ',' + str(classification[i]) + '\n')
	text_file.close()
	return 0



'''main'''
home_dir = '/home/sunyue/ROB535/'
train_image_list, test_image_list = file_read(home_dir)
classification = [randint(0, 3) for p in range(0, np.shape(test_image_list)[0])]
classes_output(classification, test_image_list)

