# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:50:54 2017

@author: yipinp
"""
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array 
import numpy as np

#user define function
def GaussianWhiteNoiseForGray(imgIn):
    global mean
    global sigma
    global dst_size
    img = np.zeros(imgIn.shape)
    gray = 255
    zu = []
    zv = []
    for i in range(0,dst_size[0]):
        for j in range(0,dst_size[1],2):
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = mean + sigma*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
            z2 = mean + sigma*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
            zu.append(z1)
            zv.append(z2)
            img[i,j] = np.clip(int(imgIn[i,j] + z1),0,gray)
            img[i,j+1] = np.clip(int(imgIn[i,j+1] + z2),0,gray)

    return img

  
#generate one batch data sets based on directory
def generate_data_sets(directory,target_size,batch_size,shuffle_enable,phase):
    #scan directory and get the images
    x_datagen = ImageDataGenerator();
    y_datagen = ImageDataGenerator(preprocessing_function= GaussianWhiteNoiseForGray);
    if phase == "TRAIN":
        train_x_generator = x_datagen.flow_from_directory(directory,target_size,batch_size = batch_size,shuffle=shuffle_enable,class_mode=None) 
        train_y_generator = y_datagen.flow_from_directory(directory,target_size,batch_size = batch_size,shuffle=shuffle_enable,class_mode=None) 
        train_generator = zip(train_x_generator,train_y_generator)  
    else :
        test_x_generator = x_datagen.flow_from_directory(directory,target_size,batch_size = batch_size,shuffle=shuffle_enable,class_mode=None) 



