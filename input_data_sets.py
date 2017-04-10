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

def scan_image_directories(directory_path):
    