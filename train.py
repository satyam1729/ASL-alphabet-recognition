#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:09:30 2018

@author: satyam
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import Input,Dense,Flatten
#import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras import applications

categories=36
train_data_dir="dataset"
img_width=386
img_height=226


inp=Input(shape=(img_width,img_height,3))#can try with conv layers before inception layer or resized version of same image
base_model=applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(img_width,img_height,3), pooling=None, classes=1000)
#print(base_model.summary())


for l in base_model.layers:
    l.trainable = False
y=base_model(inp)
x = Flatten(name='flatten')(y)
x = Dense(4096, activation='relu', name='fc1')(x)
#x = Dense(4096, activation='relu', name='fc2')(x)
output = Dense(categories, activation='softmax', name='predictions')(x)

model=Model(inputs=inp,outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_crossentropy'])
print(model.summary())


train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)


train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_width,img_height), 
class_mode = "categorical")
model.fit_generator(train_generator)