#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:09:30 2018

@author: satyam
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import Input,Dense,Flatten
from keras.models import Sequential, Model 
from keras import applications
inp=Input(shape=(386,226,3))#can try with conv layers before inception layer or resized version of same image
categories=36
base_model=applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(386,226,3), pooling=None, classes=1000)
#print(base_model.summary())
for l in base_model.layers:
    l.trainable = False
y=base_model(inp)
x = Flatten(name='flatten')(y)
x = Dense(4096, activation='relu', name='fc1')(x)
#x = Dense(4096, activation='relu', name='fc2')(x)
output = Dense(36, activation='softmax', name='predictions')(x)

model=Model(inputs=inp,outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrices=['categorical_crossentropy'])
print(model.summary())