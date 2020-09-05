# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:18:35 2020

@author: sj
"""

from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import concatenate, Dense, Dropout, Flatten, Add, SpatialDropout2D, Conv3D
from keras.layers import Conv2D, MaxPooling2D, Input, Activation,AveragePooling2D,BatchNormalization
from keras.layers import MaxPooling3D, AveragePooling3D,Conv2DTranspose,Reshape
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.initializers import he_normal, RandomNormal
from keras.layers import multiply, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers.core import Reshape, Dropout

def wcrn_recon(band, ncla1):
    input1 = Input(shape=(5,5,band))

    # define network
    conv0x = Conv2D(64,kernel_size=(1,1),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(64,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(128,kernel_size=(1,1),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
#    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    #
    bn_de1 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    dconv1 = Conv2DTranspose(128, kernel_size=(1,1), padding='valid')
    dconv2 = Conv2DTranspose(128, kernel_size=(1,1), padding='valid')
    dconv3 = Conv2DTranspose(128, kernel_size=(3,3), padding='valid')
    dconv4 = Conv2DTranspose(band, kernel_size=(3,3), padding='valid')

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)
    x1x = MaxPooling2D(pool_size=(5,5))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add(name='ploss')([x1,x11])
    
    x11 = Flatten()(x1)
    pre1 = fc1(x11)
    
#    x12 = dconv1(x1)
#    x12 = Activation('relu')(x12)
#    x12 = dconv2(x12)
#    x12 = Activation('relu')(x12)
#    x12 = dconv3(x12)
    
    x12 = bn_de1(x1)
    x12 = Activation('relu')(x12)
    x12 = dconv1(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv2(x12)
    x12 = Add()([x1,x12])
    x12 = dconv3(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv4(x12)
    
    model1 = Model(inputs=input1, outputs=[pre1,x12])
    model2 = Model(inputs=input1, outputs=pre1)
    return model1,model2

def resnet99_avg_recon(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    #
    dconv1 = Conv2DTranspose(64, kernel_size=(1,1), padding='valid')
    dconv2 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv3 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv4 = Conv2DTranspose(64, kernel_size=(3,3), padding='valid')
    dconv5 = Conv2DTranspose(band, kernel_size=(3,3), padding='valid')
    bn1_de = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    bn2_de = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
#    x1 = MaxPooling2D(pool_size=(2,2))(x1)
#    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    if l==2:
        x11 = bn21(x1)
        x11 = Activation('relu')(x11)
        x11 = conv21(x11)
        x11 = Activation('relu')(x11)
        x11 = conv22(x11)
        x1 = Add()([x1,x11])
    
    x1 = GlobalAveragePooling2D(name='ploss')(x1)
    pre1 = fc1(x1)
    
    #
    x12 = Reshape((1,1,64))(x1)
    x12 = dconv1(x12)
    x12 = bn1_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv2(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv3(x12)
    x12 = bn2_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv4(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv5(x12)

    model1 = Model(inputs=input1, outputs=[pre1,x12])
    model2 = Model(inputs=input1, outputs=pre1)
    return model1,model2