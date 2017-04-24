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
def image_normalization(imgIn):
    img_norm = imgIn/256.0;
    return img_norm
    
#horizontal scan first then vertical,output array is [patch_height,patch_width,channel,number]    
def next_batch(result,batch_num):
    global patch_current_x
    global patch_current_y
    global current_file_id
    global current_image
    global current_image_true
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
        print("All training set is finished, need to reset the training set now!")
        
    if current_x == 0 and current_y == 0 and current_file_id < len(result):
        image,image_true,_ = get_one_image(result[current_file_id]) 
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
    patch_height = patch_size[0]
    patch_width  = patch_size[1]
    stride = patch_stride
    image,image_true,_ = get_one_image(image_name)
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
    return np.reshape(c[0:i+1,:,:],(i+1,-1)),np.reshape(c_true[0:i+1,:,:],(i+1,-1))

"""
    convert image to fixed size, and do 3x3 matrix multiple
"""
def get_one_image(filename):
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
    
    if noise_model == 1:
        img = GaussianWhiteNoiseForGray(img_true,dst_size,noise_mean,noise_sigma)
    elif noise_model == 2:
        img = saltAndPepperForRGB(img_true,0)
    else :
        img = img_true
    
    img = image_normalization(img)
    img_true1 = image_normalization(img_true)
    return img,img_true1,img_true
    
    
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
   ---------------------------------------------------------------------------
                 Basic layers & MLP & AlexNet 
"""
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']                     
    return out_layer
       
"""
        User defined parameter
"""    
    
#global setting        
image_size =(224,224)
mini_batch_num = 10

# image anti-clockwise rotation angle in preprocess phase  
theta = 0

#flip 
flip_mode = None  # None : None, 0 : vertical flip , positive : horizontal flip, negative: horizontal and vertical
 


#add noise model in preprocess phase 
noise_model = 1  #0 : NONE, 1: Gaussian 2: salt and pepper noise
noise_mean = 0.0
noise_sigma  = 20.0
salt_percent = 0.0

#patch parameters
patch_size = (28,28)
patch_stride = 14

#image scan directory setting
training_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\training_data_set'
test_set_dir = r'C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\datasets\test_data_set'
current_file_id = 0
model_path = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\model.ckpt"
img_path = r"C:\Nvidia\my_library\visualSearch\TNR\github\denoising_dl\output.jpg"


#training_set_dir = r'/home/pyp/paper/denosing/denoising_dl/data' 
#img_path = r'/home/pyp/paper/denosing/denoising_dl/output.jpg'
#model_path = r'/home/pyp/paper/denosing/denoising_dl/model.ckpt'
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
learning_rate = 0.005
training_epochs = 30
batch_size = 1000
num_examples = 10000
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_output = 784 # denoised patch size (img shape: 28*28)
channel = 1

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

#test program         
result = scan_image_directories(training_set_dir)
result = random_image_list(result,seed)

result_test = scan_image_directories(test_set_dir)
result_test = random_image_list(result_test,seed)

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.nn.l2_loss(pred-y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        current_file_id = 0 # reset patch read index
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(result,batch_size)
            
            if batch_y == None:
                break
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            #image_recovery(image_size[0],image_size[1],patch_size[0],patch_size[1],patch_stride,batch_y)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    

    print("Start test phase for one image!")
    test_image = result_test[0]
    print("test image is:",test_image)
    patch_current_y = 0
    patch_current_x = 0
    batch_x,batch_y = get_patches_one_image(test_image)
    patch_recover,cost_test = sess.run([pred,cost],{x:batch_x,y:batch_y})
    print("Test phase: cost = ",cost_test)
    #print(sess.run(tf.reduce_max(patch_recover)),sess.run(tf.reduce_max(weights['h1'])))
    frame = image_recovery(image_size[0],image_size[1],patch_size[0],patch_size[1],patch_stride,patch_recover)
    save_path = saver.save(sess,model_path)
    golden_image = get_golden_image_show(test_image)
    plt.subplot(1,2,1)
    plt.imshow(frame,cmap='gray')
    cv2.imwrite(img_path,frame*256.0)
    plt.subplot(1,2,2)
    plt.imshow(golden_image,cmap='gray')



