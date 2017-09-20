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
import tensorflow as tf
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from sklearn import preprocessing

#from tflearn.layers.core import input_data,dropout,fully_connected
#from tflearn.layers.conv import conv2d,max_pool_2d
#from tflearn.layers.normalization import local_response_normalization


"""
    Scan one directory and find all files in the directory. return file list
"""
def scan_image_directories(directory_path):
    result = []
    for root,dirs,files in os.walk(directory_path):
        for filename in files:
            print(os.path.join(root,filename))
            #check filename for jpg,jpeg,png only
            if(os.path.splitext(filename)[1] != ".jpg" and os.path.splitext(filename)[1] != ".jpeg" and os.path.splitext(filename)[1] != ".png"):
                continue
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
            img[i,j,0] = np.clip(int(imgIn[i,j,0] + z1),0,gray)
            img[i,j+1,0] = np.clip(int(imgIn[i,j+1,0] + z2),0,gray)
            img[i,j,1] = np.clip(int(imgIn[i,j,1] + z1),0,gray)
            img[i,j+1,1] = np.clip(int(imgIn[i,j+1,1] + z2),0,gray)
            img[i,j,2] = np.clip(int(imgIn[i,j,2] + z1),0,gray)
            img[i,j+1,2] = np.clip(int(imgIn[i,j+1,2] + z2),0,gray)   
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
        
    
def GaussianWhiteNoiseForGray(imgIn,dst_size,mean,sigma):
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
            """
            img[i,j,1] = np.clip(int(img[i,j] + z1),0,gray)
            img[i,j+1,1] = np.clip(int(img[i,j+1] + z2),0,gray)
            img[i,j,2] = np.clip(int(img[i,j] + z1),0,gray)
            img[i,j+1,2] = np.clip(int(img[i,j+1] + z2),0,gray) 
            """
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
def get_normal_param(imageIn,mode):
    if mode == 0 :
        mean = np.mean(imageIn.reshape(-1))
        stddev = np.std(imageIn.reshape(-1))
        return mean,stddev
    else:
       dmin = np.min(imageIn)
       dmax = np.max(imageIn)  
       return dmin,dmax 
    
def image_normalization(imgIn,p0,p1,mode):
   # return preprocessing.scale(imgIn)  
    if mode == 0:
        #img_norm = (imgIn/255.0 - 0.5)*0.2;
        img_norm = (imgIn - p0)/p1
        #img_norm = imgIn/255.0
        return img_norm
    else:
        return (imgIn - p0)/(p1 - p0)
        
def image_renorm(frame,p0,p1,mode):
    if mode == 0:
        #frame = (frame*5.0 + 0.5)*255.0
        frame = frame * p1 + p0

    else:
        frame = frame * (p1 - p0) + p0
        
    frame = np.array(frame,dtype=int)
    frame = frame.clip(0,255)
    return frame

    
    
#horizontal scan first then vertical,output array is [patch_height,patch_width,channel,number]    
def next_batch(result,batch_num):
    global patch_current_x
    global patch_current_y
    global current_file_id
    global current_image
    global current_image_true
    global mode
    image = current_image
    image_true = current_image_true
    patch_height = patch_size[0]
    patch_width  = patch_size[1]
    num = batch_num
    stride = patch_stride
    current_x = patch_current_x
    current_y = patch_current_y
    
    if current_file_id >= len(result):
        current_file_id = 0   #reload training set if finished
       # print("All training set is finished, need to reset and shuffle the training set now!")
        random.shuffle(result)
        
    if current_x == 0 and current_y == 0 and current_file_id < len(result):
        image,image_true,_,_,_ = get_one_image(result[current_file_id]) 
        current_image = image
        current_image_true = image_true
        current_file_id = current_file_id + 1
        
        
    c = np.zeros((num,patch_height,patch_width),dtype="float32")
    c_true = np.zeros((num,patch_height,patch_width),dtype="float32")
    for i in range(num):
        a = image[current_y:min(current_y+patch_height,image.shape[0]),current_x:min(current_x+patch_width,image.shape[1])]
        b = image_true[current_y:min(current_y+patch_height,image.shape[0]),current_x:min(current_x+patch_width,image.shape[1])]
        #Fill 0 when out of picture
        c[i,:a.shape[0],:a.shape[1]] = a
        c_true[i,:a.shape[0],:a.shape[1]] = b
        #update next patch coordination
        if current_x+stride >= image.shape[1]:
            current_x = 0
            current_y += stride
        else:
            current_x += stride
            
        if current_y + stride >= image.shape[0]:
            #current_y = image.shape[0] + 1 
            current_x = 0
            current_y = 0
            break
        
    patch_current_x = current_x
    patch_current_y = current_y  
    #print("get batch number:",i,num,current_file_id,len(result),patch_current_x,patch_current_y,image.shape)
    return np.reshape(c[0:i+1,:,:],(i+1,-1)),np.reshape(c_true[0:i+1,:,:],(i+1,-1))


def get_patches_one_image(image_name):
    global patch_current_x
    global patch_current_y
    global mode
    patch_height = patch_size[0]
    patch_width  = patch_size[1]
    stride = patch_stride
    image,image_true,_,p0,p1 = get_one_image(image_name)
    height_in_patch = (image.shape[0] + patch_stride - 1)//patch_stride
    width_in_patch = (image.shape[1] + patch_stride - 1)//patch_stride   
    num = height_in_patch * width_in_patch                   
    c = np.zeros((num,patch_height,patch_width),dtype="float32")
    c_true = np.zeros((num,patch_height,patch_width),dtype="float32")
    current_x = patch_current_x
    current_y = patch_current_y
   
    for i in range(num):
        a = image[current_y:min(current_y+patch_height,image.shape[0]),current_x:min(current_x+patch_width,image.shape[1])]    
        c[i,:a.shape[0],:a.shape[1]] = a
        a_true = image_true[current_y:min(current_y+patch_height,image.shape[0]),current_x:min(current_x+patch_width,image.shape[1])]
        c_true[i,:a_true.shape[0],:a_true.shape[1]] = a_true
        if current_x+patch_stride >= image.shape[1]:
            current_x = 0
            current_y += stride
        else:
            current_x += stride
    return np.reshape(c[0:i+1,:,:],(i+1,-1)),np.reshape(c_true[0:i+1,:,:],(i+1,-1)),p0,p1,num

"""
    convert image to fixed size, and do 3x3 matrix multiple
"""
def get_one_image(filename):
    dst_size = image_size
    img_origin = cv2.imread(filename)
    global channel
    global mode
    global theta
    #convert to gray image
    if channel == 1:
        img_origin = cv2.cvtColor(img_origin,cv2.COLOR_BGR2GRAY)
    img_true = cv2.resize(img_origin,dst_size,interpolation=cv2.INTER_CUBIC)
    rows,cols = img_true.shape
    M = cv2.getRotationMatrix2D((cols/2.0,rows/2.0),theta,1.0)
    img_true = cv2.warpAffine(img_true,M,(rows,cols))
    
    if flip_mode != None :
        img_true = cv2.flip(img_true,flip_mode)    
    
    if noise_model == 1:
        img = GaussianWhiteNoiseForGray(img_true,dst_size,noise_mean,noise_sigma)
    elif noise_model == 2:
        img = saltAndPepperForRGB(img_true,0)
    else :
        img = img_true
    p0,p1 = get_normal_param(img,mode)
    img = image_normalization(img,p0,p1,mode)
    p0_true,p1_true = get_normal_param(img_true,mode)
    img_true1 = image_normalization(img_true, p0,p1,mode)
    return img,img_true1,img_true,p0,p1
    
    
    #"""
    plt.imshow(img)
    plt.show()
    print(img.shape)
    print(img_origin.shape)
    #"""

    
def get_golden_image_show(filename):
    dst_size = image_size
    img_origin = cv2.imread(filename)
    global channel
    #convert to gray image
    if channel == 1:
        img_origin = cv2.cvtColor(img_origin,cv2.COLOR_BGR2GRAY)
        
    img_true = cv2.resize(img_origin,dst_size,interpolation=cv2.INTER_CUBIC)
    rows,cols = img_true.shape
    M = cv2.getRotationMatrix2D((cols/2.0,rows/2.0),theta,1.0)
    img_true = cv2.warpAffine(img_true,M,(rows,cols))

    if flip_mode != None :
        img_true = cv2.flip(img_true,flip_mode)    
    
    return img_true
    
#Horizontal patch scan   
def image_recovery(frame_height,frame_width,patch_height,patch_width,patch_stride,patches):
    frame_width_in_patch = (frame_width + patch_stride - 1)//patch_stride
    frame_height_in_patch = (frame_height + patch_stride - 1)//patch_stride
    frame = np.ones((frame_height_in_patch*patch_height,frame_width_in_patch*patch_width)) * -1 
    for i in range(frame_height_in_patch):
        for j in range(frame_width_in_patch):
            patch_x = j * patch_stride
            patch_y = i * patch_stride        
          #  print("patch:",patch_x,patch_y,i,j,frame_height_in_patch,frame_width_in_patch,patch_stride,patches.shape,i*frame_width_in_patch+j)
            patch = patches[i*frame_width_in_patch+j,:]
            np.savetxt("patch.txt",patches,fmt="%f")

            for m in range(patch_height):
                for n in range(patch_width):
                    
                    if frame[patch_y+m][patch_x+n] < 0:  #the first patch
                        if (patch_y+m < frame_height) and (patch_x+n < frame_width) :
                            frame[patch_y+m][patch_x+n] = patch[m*patch_width + n]
                    else :
                        if (patch_y+m < frame_height) and (patch_x+n < frame_width) :
                            frame[patch_y+m][patch_x+n] += patch[m*patch_width + n]
                            frame[patch_y+m][patch_x+n] /=2.0;
                    
             
    np.savetxt("frame.txt",frame[0:frame_height,0:frame_width],fmt="%f")    
    return frame[0:frame_height,0:frame_width]



"""
    image quality evalution psnr/ssim
    
    mse = ((image-image_golden)**2).mean()
    if mode == "PSNR":
        cost = 10*np.log10((255**2)/mse)
    elif mode == "SSIM":
        cost = ssim(image,image_golden)    
    
    
"""
def evaluate_image(mode,image,image_golden):

    if mode == "PSNR":
        cost = psnr(image,image_golden,255.0)
    elif mode == "SSIM": 
        cost = ssim(image,image_golden,data_range=255.0)
    
    return cost



    
"""    
   ---------------------------------------------------------------------------
                 Basic layers & MLP & AlexNet 
"""
# Create model
def multilayer_perceptron(x, weights, biases,activate):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    if activate =="Relu":
        layer_1 = tf.nn.relu(layer_1)
    elif activate =="Sigmoid":
        layer_1 = tf.sigmoid(layer_1)
    else:
        layer_1 = tf.tanh(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    if activate =="Relu":
        layer_2 = tf.nn.relu(layer_2)
    elif activate =="Sigmoid":
        layer_2 = tf.sigmoid(layer_2)
    else:
        layer_2 = tf.tanh(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    if activate =="Relu":
        layer_3 = tf.nn.relu(layer_3)
    elif activate =="Sigmoid":
        layer_3 = tf.sigmoid(layer_3)
    else:
        layer_3 = tf.tanh(layer_3)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
                     
    return out_layer
    
    
    
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.tanh(x)
    return tf.nn.relu(x)
    
    
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def conv2d_batch(x, W, b,phase,strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides,strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.contrib.layers.batch_norm(x, 
                                     center=True, scale=True, 
                                     is_training=phase
                                   )
    return tf.tanh(x)
    return tf.nn.relu(x)
    
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
    
    
# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['h1'], biases['b1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['h2'], biases['b2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['h3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['h3']), biases['b3'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
    
def conv_net_batch(x, weights, biases,phase):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d_batch(x, weights['h1'], biases['b1'],phase)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d_batch(conv1, weights['h2'], biases['b2'],phase)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['h3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['h3']), biases['b3'])
    
    fc1 = tf.contrib.layers.batch_norm(fc1, 
                                       center=True, scale=True, 
                                       is_training=phase
                                     )
    fc1 = tf.reshape(fc1, [-1, weights['out'].get_shape().as_list()[0]])
    #fc1 = tf.nn.relu(fc1)
    fc1 = tf.tanh(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def alex_net(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2)
    # 归一化层
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # 卷积
    conv2 = conv2d( norm1, _weights['wc2'], _biases['bc2'])
    # 下采样
    pool2 = max_pool('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # 卷积
    conv3 = conv2d(norm2, _weights['wc3'], _biases['bc3'])
    # 下采样
    pool3 = max_pool('pool3', conv3, k=2)
    # 归一化
    norm3 = norm('norm3', pool3, lsize=4)
    # Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # 全连接层，先把特征图转为向量
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
    # 全连接层
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # 网络输出层
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out
    
       
"""
        User defined parameter
"""    
    
#global setting        
image_size =(224,224)

# image anti-clockwise rotation angle in preprocess phase  
theta = 0

#flip 
flip_mode = None  # None : None, 0 : vertical flip , positive : horizontal flip, negative: horizontal and vertical
 


#add noise model in preprocess phase 
noise_model = 1  #0 : NONE, 1: Gaussian 2: salt and pepper noise
noise_mean = 0.0
noise_sigma  = 20.0
salt_percent = 0.0

#board


#image scan directory setting
#training_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\training_data_set'
#test_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\test_data_set'
training_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\overfit'
test_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\overfit'
current_file_id = 0
model_path = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\models"
model_name = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\models\model.ckpt"
logs_path = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\board'
img_path = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\output"
log_file  = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\output.log"
"""
training_set_dir = r'/home/pyp/paper/denosing/denoising_dl/training_data' 
test_set_dir = r'/home/pyp/paper/denosing/denoising_dl/test_data' 
img_path = r'/home/pyp/paper/denosing/denoising_dl/output'
model_path = r'/home/pyp/paper/denosing/denoising_dl/models'
model_name = r'/home/pyp/paper/denosing/denoising_dl/models/model.ckpt'
logs_path = r'/home/pyp/paper/denosing/denoising_dl/board'
log_file  = r'/home/pyp/paper/denosing/denoising_dl/output.log'
"""
#random training sets
seed = 0    #fixed order with fixed seed 

"""
    Internal variable , no setting by user
"""
#internal global variable, no setting by user
patch_current_x = 0
patch_current_y = 0
current_image = None
current_image_true = None


"""
 ---------------------------------------------------------
                  MLP control parameters
"""
tf.reset_default_graph()
learning_period = 30
learning_ratio = 1.0
training_epochs = 100
batch_size = 128
num_examples = 6000
display_step = 1
threshold_adjust = 0.90
early_termination_threshold = 1/100000
mode = 1 #mean,stddev, 1: min,max
#continuous traing or training from scatch
#training_mode = "test_only" # continuous,test_only
training_mode = "continuous"
reset_learning_rate_enable = 0
reset_learning_rate = 0.00001  #when rerun the epoch, if need to reset the learning rate
learning_rate_threshold = 0.001 #not below the value for learning rate
channel = 1
save_step = 20
prev_cost = 0
network = "ALEXNET"

#patch parameters

if network == "MLP" :
    patch_size = (14,14)
    patch_stride = 14
    n_input = patch_size[0]*patch_size[1] # MNIST data input (img shape: 28*28)
    n_output = patch_size[0]*patch_size[1] # denoised patch size (img shape: 28*28)
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_hidden_3 = 256 # 3nd layer number of features
    dropout = 0.000001 # Dropout, probability to keep units
    active_mode = "Relu" #Sigmoid,Tanh
    #Relu use He weight initialization 
    if active_mode == "Relu": 
        weights = {
            'h1': tf.Variable(np.random.randn(n_input,n_hidden_1)/np.sqrt(n_input/2),dtype="float32"),
            'h2': tf.Variable(np.random.randn(n_hidden_1,n_hidden_2)/np.sqrt(n_hidden_1/2),dtype="float32"),
            'h3': tf.Variable(np.random.randn(n_hidden_2,n_hidden_3)/np.sqrt(n_hidden_2/2),dtype="float32"),
            'out': tf.Variable(np.random.randn(n_hidden_3,n_output)/np.sqrt(n_hidden_3/2),dtype="float32")
            }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),                                    
            'out': tf.Variable(tf.random_normal([n_output]))
        }  
    elif active_mode == "Sigmoid": #sqrt(6/(fan_in+fan_out))
        weights = {
            'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1],minval=-np.sqrt(6/(n_input+n_hidden_1)),maxval=np.sqrt(6/(n_input+n_hidden_1)),dtype=tf.float32)),
            'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2],minval=-np.sqrt(6/(n_hidden_2+n_hidden_1)),maxval=np.sqrt(6/(n_hidden_2+n_hidden_1)),dtype=tf.float32)),
            'h3': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3],minval=-np.sqrt(6/(n_hidden_2+n_hidden_3)),maxval=np.sqrt(6/(n_hidden_2+n_hidden_3)),dtype=tf.float32)),
            'out': tf.Variable(tf.random_uniform([n_hidden_3, n_output],minval=-np.sqrt(6/(n_hidden_3+n_output)),maxval=np.sqrt(6/(n_hidden_3+n_output)),dtype=tf.float32))
            }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),                                    
            'out': tf.Variable(tf.random_normal([n_output]))
        } 
    else: #tanh set 4*sqrt(6/(fan_in+fan_out)) 
        weights = {
            'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1],minval=-4.0*np.sqrt(6/(n_input+n_hidden_1)),maxval=4.0*np.sqrt(6/(n_input+n_hidden_1)),dtype=tf.float32)),
            'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2],minval=-4.0*np.sqrt(6/(n_hidden_2+n_hidden_1)),maxval=4.0*np.sqrt(6/(n_hidden_2+n_hidden_1)),dtype=tf.float32)),
            'h3': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3],minval=-4.0*np.sqrt(6/(n_hidden_2+n_hidden_3)),maxval=4.0*np.sqrt(6/(n_hidden_2+n_hidden_3)),dtype=tf.float32)),
            'out': tf.Variable(tf.random_uniform([n_hidden_3, n_output],minval=-4.0*np.sqrt(6/(n_hidden_3+n_output)),maxval=4.0*np.sqrt(6/(n_hidden_3+n_output)),dtype=tf.float32))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),                                    
            'out': tf.Variable(tf.random_normal([n_output]))
        } 
elif network == "CNN" or network == "CNNBATCH": #with good weight initialization and batch normalization layer
    patch_size = (28,28)
    patch_stride = 14
    n_input = patch_size[0]*patch_size[1] # MNIST data input (img shape: 28*28)
    n_output = patch_size[0]*patch_size[1] # denoised patch size (img shape: 28*28)
    dropout = 0.5 # Dropout, probability to keep units
    W_init = tf.contrib.layers.xavier_initializer_conv2d()      
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'h1': tf.get_variable('h1', [5, 5, 1, 32], initializer=W_init),
        # 5x5 conv, 32 inputs, 64 outputs
        'h2': tf.get_variable('h2', [5, 5, 32, 64], initializer=W_init),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'h3': tf.get_variable('h3', [7*7*64, 1024], initializer=W_init),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.get_variable('out', [1024, n_output], initializer=W_init)
     }

    biases = {
        'b1': tf.Variable(tf.random_normal([32])),
        'b2': tf.Variable(tf.random_normal([64])),
        'b3': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }  

elif network == "CNNBATCH1": #no good weight optimization with batch normalization layer
    patch_size = (28,28)
    patch_stride = 14
    n_input = patch_size[0]*patch_size[1] # MNIST data input (img shape: 28*28)
    n_output = patch_size[0]*patch_size[1] # denoised patch size (img shape: 28*28)   
    
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'h1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'h2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'h3': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_output]))
     }

    biases = {
        'b1': tf.Variable(tf.random_normal([32])),
        'b2': tf.Variable(tf.random_normal([64])),
        'b3': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }          
    
    batchN_scale = {
        'b1': tf.Variable(tf.ones([32])),
        'b2': tf.Variable(tf.ones([64])),
        'b3': tf.Variable(tf.ones([1024]))
     }

    batchN_beta = {
        'b1': tf.Variable(tf.ones([32])),
        'b2': tf.Variable(tf.ones([64])),
        'b3': tf.Variable(tf.ones([1024]))
     }

elif network == "ALEXNET":
    patch_size = (28,28)
    patch_stride = 14
    n_input = patch_size[0]*patch_size[1] # MNIST data input (img shape: 28*28)
    n_output = patch_size[0]*patch_size[1] # denoised patch size (img shape: 28*28) 
    dropout = 0.5 # Dropout, probability to keep units
    W_init = tf.contrib.layers.xavier_initializer_conv2d()  
    
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.get_variable('wc1', [3, 3, 1, 64], initializer=W_init),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.get_variable('wc2', [3, 3, 64, 128], initializer=W_init),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wc3': tf.get_variable('wc3', [3, 3, 128, 256], initializer=W_init),
        'wd1': tf.get_variable('wd1', [4*4*256, 1024], initializer=W_init),
        'wd2': tf.get_variable('wd2', [1024, 1024], initializer=W_init),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.get_variable('out', [1024, n_output], initializer=W_init)
     }

    """
    weights = {
     'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
     'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
     'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
     'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
     'wd2': tf.Variable(tf.random_normal([1024, 1024])),
     'out': tf.Variable(tf.random_normal([1024, n_output]))
    }
    """
    biases = {
     'bc1': tf.Variable(tf.random_normal([64])),
     'bc2': tf.Variable(tf.random_normal([128])),
     'bc3': tf.Variable(tf.random_normal([256])),
     'bd1': tf.Variable(tf.random_normal([1024])),
     'bd2': tf.Variable(tf.random_normal([1024])),
     'out': tf.Variable(tf.random_normal([n_output]))
    }
    

#Vriable defines which can be save/restore by saver
learning_rate = tf.Variable(0.001,dtype="float",name="learning_rate")
    
seed = time.time()
print("random seed is :", seed)
tf.set_random_seed(seed)
random.seed(seed)

#set weight initialization range for different activate
weight_init_min = 0.0
weight_init_max = 2.0  


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
phase = tf.placeholder(tf.bool, name='phase')
#lrate = tf.placeholder("float",[1])

#test program         
result = scan_image_directories(training_set_dir)
result = random_image_list(result,seed)

result_test = scan_image_directories(test_set_dir)
result_test = random_image_list(result_test,seed)

# Construct model
if network == "MLP":
    pred = multilayer_perceptron(x, weights, biases,active_mode)
elif network == "CNN":
    pred = conv_net(x, weights, biases, keep_prob)
elif network == "CNNBATCH" or network == "CNNBATCH1":
    pred = conv_net_batch(x, weights, biases,phase)
elif network == "ALEXNET":
    pred = alex_net(x, weights, biases, keep_prob)
    
# Define loss and optimizer
cost = tf.nn.l2_loss(pred-y)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
saver = tf.train.Saver()

#tensorboard
#tf.summary.histogram("weight1",weights['h1'])
#tf.summary.histogram("weight2",weights['h2'])
#tf.summary.histogram("weight3",weights['h3'])
#tf.summary.histogram("weightout",weights['out'])
#tf.summary.histogram("bias1",biases['b1'])
#tf.summary.histogram("bias2",biases['b2'])
#tf.summary.histogram("bias3",biases['b3'])
#tf.summary.histogram("biasout",biases['out'])
tf.summary.scalar("loss",cost)
merged_summary_op = tf.summary.merge_all()

f_log = open(log_file,"w")

# Launch the graph for training phase
if training_mode != "test_only":
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        if training_mode == "continuous" :
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("continuous mode is enabled, the cp is :",ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
                if reset_learning_rate_enable == 1:
                    print("Reset learning rate after restore when the learning rate is too small for converge!")
                    sess.run(tf.assign(learning_rate,reset_learning_rate))
                    
        # Training cycle
        print("Start training!")        
        for epoch in range(training_epochs):
            avg_cost = 0.
            current_file_id = 0 # reset patch read index
            total_batch = int(num_examples/batch_size)
            #data augment for theta rotation
            if epoch % learning_period == learning_period - 1 :
              #  theta = np.random.uniform(-180,180)
                print("epoch:",epoch+1," the theta is:",theta)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(result,batch_size)
            
                if batch_y is None:
                    break
                # Run optimization op (backprop) and cost op (to get loss value)
                if network == "MLP" or network == "CNNBATCH" or network == "CNNBATCH1" :
                    _, c,summary = sess.run([optimizer,cost,merged_summary_op], feed_dict={x: batch_x,
                                                          y: batch_y,phase:True,keep_prob:dropout
                                                          })
                elif network == "CNN"  or network == "ALEXNET":
                    _, c,summary = sess.run([optimizer,cost,merged_summary_op], feed_dict={x: batch_x,
                                                          y: batch_y,keep_prob:dropout
                                                          })                    
                
               #image_recovery(image_size[0],image_size[1],patch_size[0],patch_size[1],patch_stride,batch_y)
               # Compute average loss
                summary_writer.add_summary(summary,epoch*total_batch+i)                
                avg_cost += c/num_examples   
                # Display logs per epoch step
            if epoch % learning_period == 0:
                print("current learning rate is :",sess.run(learning_rate))
                prev_cost = avg_cost
            """    
            if avg_cost > prev_cost*threshold_adjust  and epoch % learning_period == learning_period - 1:
                 if abs(prev_cost - avg_cost) < early_termination_threshold*prev_cost:
                     print("Early termination!",prev_cost,avg_cost)
                     #break
                 if sess.run(learning_rate) > learning_rate_threshold :
                     sess.run(tf.assign(learning_rate,learning_rate *learning_ratio))
                     print("Adjust learning rate to ",sess.run(learning_rate))
            """            
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost),",weight_out:",sess.run(tf.nn.moments(tf.reshape(weights['out'],[-1]),axes=[0])),",bias out:",sess.run(tf.nn.moments(tf.reshape(biases['out'],[-1]),axes=[0])))
                
            if (epoch + 1)% save_step == 0 :
                print("Save model in epoch ",epoch+1, " learning rate is:",sess.run(learning_rate)) 
                saver.save(sess,model_name,global_step = epoch + 1)

        #save the last step   
        print("Save model in epoch ",epoch+1, " learning rate is:",sess.run(learning_rate)) 
        saver.save(sess,model_name,global_step = epoch+1)   
        print("Training weights out :",sess.run(weights['out']), "bias out:",sess.run(biases['out']))                
        print("Optimization Finished!")
    
#Test phase    
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("Start test phase for one image!")
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Test mode is enabled, load the cp is :",ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        print("Restore model and learning rate is:",sess.run(learning_rate))
    
    #reset theta to zero in test phase
    theta = 0   
    for i in range(len(result_test)):
        test_image = result_test[i]
        print("test image is:",test_image)
        f_log.write("test image is %s\n" %(test_image))
        patch_current_y = 0
        patch_current_x = 0
        batch_x,batch_y,p0,p1,patch_num= get_patches_one_image(test_image)
        if network == "MLP" or network == "CNNBATCH":
            patch_recover,cost_test = sess.run([pred,cost],{x:batch_x,y:batch_y,phase:False,keep_prob:1.})
        elif network == "CNN" or network=="ALEXNET":
            patch_recover,cost_test = sess.run([pred,cost],{x:batch_x,y:batch_y,keep_prob: 1.})
        #print(patch_recover)
        print("Test phase: cost = ", cost_test/patch_num)
        f_log.write("Test phase: cost = %f" %(cost_test/patch_num))
        #print("Test weights out :",sess.run(weights['out']), "bias out:",sess.run(biases['out']))
        #print(sess.run(tf.reduce_max(patch_recover)),sess.run(tf.reduce_max(weights['h1'])))
        frame_denoise = image_recovery(image_size[0],image_size[1],patch_size[0],patch_size[1],patch_stride,patch_recover)
        frame_denoise = image_renorm(frame_denoise,p0,p1,mode)
        frame_noise = image_recovery(image_size[0],image_size[1],patch_size[0],patch_size[1],patch_stride,batch_x)
        frame_noise = image_renorm(frame_noise,p0,p1,mode)
        golden_image = get_golden_image_show(test_image)
        
        #Call BM3D as the denoise reference
        
        cv2.imwrite(img_path+str(i)+".jpg",frame_denoise)
        cv2.imwrite(img_path+str(i)+"_golden.jpg",golden_image)        
        cv2.imwrite(img_path+str(i)+"_noise.jpg",frame_noise)
        #print("the real frame cost:",sess.run(tf.nn.l2_loss(frame-golden_image)))
        plt.subplot(1,2,1)       
        plt.imshow(frame_denoise,cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(golden_image,cmap='gray')
        plt.show()
        
        #print image quality
        noise_psnr = evaluate_image("PSNR",golden_image.astype(np.uint8),frame_noise.astype(np.uint8))
        noise_ssim = evaluate_image("SSIM",golden_image.astype(np.uint8),frame_noise.astype(np.uint8))
        denoise_psnr = evaluate_image("PSNR",golden_image.astype(np.uint8),frame_denoise.astype(np.uint8))
        denoise_ssim = evaluate_image("SSIM",golden_image.astype(np.uint8),frame_denoise.astype(np.uint8))
        print("After denoising, the psnr is from %f to %f, ssim from %f to %f :" %(noise_psnr,denoise_psnr,noise_ssim,denoise_ssim))

    f_log.close()

