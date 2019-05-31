import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

#def unet(pretrained_weights = None,input_size = (512,512,512,3)):
def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    print('input_shape',input_shape)
    inputs = Input(input_shape)
    #print('input_dim',inputs._dim)
    print('input',inputs)
    rank=tf.rank(inputs,name=None)
    print('rank',rank)
    #if tf.dims(inputs,name=None,out_type=tf.int32) is not None:
    #if inputs.shape is not None:
    #if tf.rank(inputs,name=None)!=0:   
    #    inputs=np.moveaxis(inputs,-1,1)
    #    print('reshaped',inputs.shape)
    #else:
    #    inputs=Input(input_shape)
    #inputs= tf.convert_to_tensor(inputs, dtype=tf.float32)
    #inputs=tf.reshape(tf.convert_to_tensor((inputs,(inputs.shape[0],inputs.shape[4],inputs.shape[3],inputs.shape[2],inputs.shape[1])),dtype=tf.float32))
    #inputs=tf.reshape(tf.convert_to_tensor((inputs,(tf.shape(inputs)[0],tf.shape(inputs)[4],tf.shape(inputs)[3],tf.shape(inputs)[2],tf.shape(inputs)[1])),dtype=tf.int32))
    #print('reshaped',inputs.shape)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(inputs)
    print('conv1',conv1.shape)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv1)
    print('conv_1',conv1.shape)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2),dim_ordering='tf')(conv1)
    print('pool1',pool1.shape)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(pool1)
    print('conv2',conv2.shape)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv2)
    print('conv_2',conv2.shape)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2),dim_ordering='tf')(conv2)
    print('pool2',pool2.shape)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(pool2)
    print('conv3',conv3.shape)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv3)
    print('conv_3',conv3.shape)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2),dim_ordering='tf')(conv3)
    print('pool3',pool3.shape)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(pool3)
    print('conv4',conv4.shape)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv4)
    print('conv_4',conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)
    print('drop4',drop4.shape)
    
    up5 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(UpSampling3D(size = (2,2,2),dim_ordering='tf')(drop4))
    #up5=Conv3DTranspose(256, 2, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    print('up5',up5.shape)
    merge5 = concatenate([conv3,up5])
    print('merge5',merge5.shape)
    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(merge5)
    print('conv5',conv5.shape)
    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv5)
    print('conv_5',conv5.shape)
    drop5=Dropout(0.5)(conv5)
    print('drop5',drop5.shape)
    up6 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(UpSampling3D(size = (2,2,2),dim_ordering='tf')(drop5))
    print('up6',up6.shape)
    merge6 = concatenate([conv2,up6])
    print('merge6',merge6.shape)
    conv6 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(merge6)
    print('conv6',conv6.shape)
    conv6 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv6)
    print('conv_6',conv6.shape)
    up7 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(UpSampling3D(size = (2,2,2),dim_ordering='tf')(conv6))
    print('up7',up7.shape)
    merge7 = concatenate([conv1,up7])
    print('merge7',merge7.shape)
    conv7 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(merge7)
    print('conv7',conv7.shape)
    conv7 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv7)
    print('conv_7',conv7.shape)
    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = concatenate([conv1,up9], axis = 3)
    #conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv8 = Conv3D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dim_ordering='tf')(conv7)
    print('conv8',conv8.shape)
    conv9 = Conv3D(1, 1, activation = 'sigmoid',dim_ordering='tf')(conv8)
    print('conv9',conv9.shape)
    model = Model(input = inputs, output = conv9)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()
    #config["model_file"] = model.save('/home/vjain/notebooks/3DUnetCNN/brats/tumor_model.h5')

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)

    return model


