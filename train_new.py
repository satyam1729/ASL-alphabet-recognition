#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:09:30 2018

@author: satyam
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import Conv2D,Input,Dense,MaxPooling2D,Flatten,BatchNormalization,Activation
#import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model 
from keras import applications
from keras.callbacks import ModelCheckpoint

categories=29
train_data_dir="train_dataset"
validate_data_dir = "validation_dataset"
img_width=257
img_height=150

model = Sequential()
model.add(Conv2D(
    32, (3, 3), input_shape=(img_width, img_height, 3)
))
model.add(BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(categories, activation='softmax', name='predictions'))

# inp=Input(shape=(img_width,img_height,3))#can try with conv layers before inception layer or resized version of same image
# base_model=applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(img_width,img_height,3), pooling=None, classes=1000)
# #print(base_model.summary())
# c1=Conv2D(64, (5, 5), padding='same', activation='relu')(inp)
# p1=MaxPooling2D((3, 3), strides=(1, 1), padding='same')(c1)
# b1=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(p1)
# p=Conv2D(3, (1, 1), padding='same', activation='relu')(b1)
# y=base_model(p)
# x = Flatten(name='flatten')(y)
# x = Dense(4096, activation='relu', name='fc1')(x)
# #x = Dense(4096, activation='relu', name='fc2')(x)
# output = Dense(categories, activation='softmax', name='predictions')(x)

# model=Model(inputs=inp,outputs=output)
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_crossentropy', 'accuracy'])
# print(model.summary())


train_datagen = ImageDataGenerator(
featurewise_center=True,
featurewise_std_normalization=True,
rescale = 1./255,
horizontal_flip = True,
zoom_range = 0.2,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_width,img_height),
batch_size = 16,
class_mode = "categorical")

validation_generator = train_datagen.flow_from_directory(
validate_data_dir,
target_size = (img_width,img_height),
batch_size = 16,
class_mode = "categorical"
)

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.fit_generator(train_generator, validation_data=validation_generator, epochs=5, callbacks=callbacks_list, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
