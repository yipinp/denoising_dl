# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:21:57 2017

@author: yipinp

The package will do the data set preparation
including:
      (1) crop the image size to fixed image size and support parameter setting(256x256)
      (2) only support RGB 3 channel or Gray  channel, others will convert to RGB or gray
      (3) support image rotation for better training set generation.
      (4) Add preprocessing step such as add noise/fog and so on.
      (5) normalization the input image range [0,255] to [0,1] to light independent. 
      (6) read image to numpy.matrix.
      (7) generate patch sets based on (patch size, stride, number)
      
"""
# -*- coding: utf-8 -*- 
import os
import os.path


def scan_image_directories(directory_path):
    for root,dirs,files in os.walk(directory_path):
        for filename in files:
            print(os.path.join(root,filename))


training_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets'            
scan_image_directories(training_set_dir)
