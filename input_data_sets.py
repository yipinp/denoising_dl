# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:21:57 2017

@author: yipinp

The package will do the data set preparation
including:
      (1) crop the image size to fixed image size and support parameter setting(256x256)
      (2) only support RGB 3 channel or Gray  channel, others will convert to RGB or gray(to Image, ignore it)
      (3) support image rotation for better training set generation.(need to provide theta angle) and do scaler to fixed size
      (4) Add preprocessing step such as add noise/fog and so on.
      (5) normalization the input image range [0,255] to [0,1] for light independent.(ndarray) 
      (6) generate patch sets based on (patch size, stride, number)
      
"""
# -*- coding: utf-8 -*- 
import os
import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

"""
    Scan one directory and find all files in the directory. return file list
"""
def scan_image_directories(directory_path):
    result = []
    for root,dirs,files in os.walk(directory_path):
        for filename in files:
            #print(os.path.join(root,filename))
            result.append(os.path.join(root,filename))
    return result
    

def random_image_list(filelist,seed):
    random.seed(seed)
    new_file_list =[]
    for n in random.sample(range(len(filelist)),len(filelist)):
        new_file_list.append(filelist[n])
    
    return new_file_list
    
    
    
    
"""
        Add gaussian noise or salt peper noise
"""    
def GaussianWhiteNoiseForRGB(imgIn,dst_size,mean,sigma):
    img = imgIn
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
            img[i,j,0] = np.clip(int(img[i,j,0] + z1),0,gray)
            img[i,j+1,0] = np.clip(int(img[i,j+1,0] + z2),0,gray)
            img[i,j,1] = np.clip(int(img[i,j,1] + z1),0,gray)
            img[i,j+1,1] = np.clip(int(img[i,j+1,1] + z2),0,gray)
            img[i,j,2] = np.clip(int(img[i,j,2] + z1),0,gray)
            img[i,j+1,2] = np.clip(int(img[i,j+1,2] + z2),0,gray)   
    return img
    
    
def saltAndPepperForRGB(img,percetage):
        percetage = salt_percent 
        dst_size = image_size
        width = dst_size[1]
        height = dst_size[0]
        NoiseNum = int(width*height*percetage)
        for i in range(NoiseNum):
            randx    = np.random.randint(0,width-1)
            randy    = np.random.randint(0,height-1)
            #print img.shape,randx,randy
            if np.random.randint(0,1):
                img[randy,randx,0] =  0
                img[randy,randx,1] =  0
                img[randy,randx,2] =  0
            else :
                img[randy,randx,0] =  255
                img[randy,randx,1] =  255
                img[randy,randx,2] =  255
                
        return img
        
        
        
"""
    read image to numpy matrix and Normalization range to [0,1],
    generate patch based on patch_size,stride,number

"""
def image_normalization(imgIn):
    img_norm = imgIn/255.0;
    return img_norm
    
#horizontal scan first then vertical,output array is [patch_height,patch_width,channel,number]    
def get_next_patches(result,batch_num):
    global patch_current_x
    global patch_current_y
    global current_file_id
    global current_image
    image = current_image
    patch_height = patch_size[0]
    patch_width  = patch_size[1]
    num = batch_num
    stride = patch_stride
    current_x = patch_current_x
    current_y = patch_current_y
    print(current_x,current_y)
    if current_x == 0 and current_y == 0 and current_file_id < len(result):
        image = get_one_image(result[current_file_id]) 
        current_image = image
        current_file_id = current_file_id + 1
        
    if current_file_id >= len(result):
        return None
        
    c = np.zeros((patch_height,patch_width,image.shape[2],num),dtype="float32")
    for i in range(num):
        a = image[current_y:min(current_y+patch_height,image.shape[0]),current_x:min(current_x+patch_width,image.shape[1]),:]
        #Fill 0 when out of picture
        c[:a.shape[0],:a.shape[1],:a.shape[2],i] = a

        #update next patch coordination
        if current_x+patch_width >= image.shape[1]:
            current_x = 0
            current_y += stride
        else:
            current_x += stride
            
        if current_y + patch_height >= image.shape[0]:
            current_y = image.shape[0] + 1 
            break
        
    patch_current_x = current_x
    patch_current_y = current_y  
    return c[:,:,:,0:i+1]



"""
    convert image to fixed size, and do 3x3 matrix multiple
"""
def get_one_image(filename):
    dst_size = image_size
    img_origin = cv2.imread(filename)
    img = cv2.resize(img_origin,dst_size,interpolation=cv2.INTER_CUBIC)
    rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2.0,rows/2.0),theta,1.0)
    img = cv2.warpAffine(img,M,(rows,cols))
    if noise_model == 1:
        img = GaussianWhiteNoiseForRGB(img,dst_size,noise_mean,noise_sigma)
    elif noise_model == 2:
        img = saltAndPepperForRGB(img,0)
    
    img = image_normalization(img)
    return img
    
    
    #"""
    plt.imshow(img)
    plt.show()
    print(img.shape)
    print(img_origin.shape)
    #"""

    
"""
        User defined parameter
"""    
    
#global setting        
image_size =(224,224)
mini_batch_num = 10

# image anti-clockwise rotation angle in preprocess phase  
theta = 0
#add noise model in preprocess phase 
noise_model = 0  #0 : NONE, 1: Gaussian 2: salt and pepper noise
noise_mean = 0.0
noise_sigma  = 0.0
salt_percent = 0.0

#patch parameters
patch_size = (5,5)
patch_stride = 220

#image scan directory setting
training_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets' 
current_file_id = 0

#random training sets
seed = 0    #fixed order with fixed seed 

"""
    Internal variable , no setting by user
"""
#internal global variable, no setting by user
patch_current_x = 0
patch_current_y = 0
current_image = None


#test program         
result = scan_image_directories(training_set_dir)
result = random_image_list(result,seed)
for i in range(10):
    get_next_patches(result,mini_batch_num)

