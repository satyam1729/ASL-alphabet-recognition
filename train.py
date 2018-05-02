#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:09:30 2018
"""
import os
import numpy as np
from tensorflow.python.keras.backend import print_tensor
from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.layers import Flatten, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16

os.environ['KERAS_BACKEND'] = 'tensorflow'
categories = 29
train_data_dir = "train_dataset"
validate_data_dir = "validation_dataset"
aug_train_data_dir = "aug_train_data_dir"
aug_validate_data_dir = "aug_validate_data_dir"
train_prefix = "train"
validate_prefix = "validate"
image_prefix = "jpg"
img_width = 150
img_height = 150
train_set_mean = np.array([132.59479491, 127.57392624, 131.1389212])
train_set_std = np.array([57.95601941, 65.03537769, 67.78074134])

if not os.path.exists(aug_train_data_dir):
    os.makedirs(aug_train_data_dir)
if not os.path.exists(aug_validate_data_dir):
    os.makedirs(aug_validate_data_dir)

# model = Sequential()
# model.add(Conv2D(
#     32, (3, 3), input_shape=(img_width, img_height, 3)
# ))
# model.add(BatchNormalization(
#     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
# ))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(BatchNormalization(
#     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
# ))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3)))
# model.add(BatchNormalization(
#     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
# ))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(categories, activation='softmax', name='predictions'))

vgg = VGG16(
        include_top=False, weights='imagenet',
        input_tensor=None, input_shape=(img_width, img_height, 3),
        pooling=None, classes=categories
    )
for layer in vgg.layers[:200]:
    layer.trainable = False

model1 = Sequential()

model1.add(vgg)
model1.add(Flatten())
model1.add(Dense(256))
model1.add(BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
))
model1.add(Activation('relu'))
model1.add(Dense(categories, activation='softmax', name='predictions'))


def test_metric(y_true, y_pred):
    print_tensor(y_true)
    print_tensor(y_pred)
    return y_true - y_pred


model1.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=[
        'categorical_accuracy',
        'categorical_crossentropy',
    ]
)

print(model1.summary())

train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=60,
        zoom_range=0.2
    )

train_datagen.mean = train_set_mean
train_datagen.std = train_set_std

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode="categorical",
        save_to_dir=aug_train_data_dir,
        save_prefix=train_prefix,
        save_format=image_prefix
    )

print(train_generator.class_indices)

validation_generator = train_datagen.flow_from_directory(
        validate_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode="categorical",
        save_to_dir=aug_validate_data_dir,
        save_prefix=validate_prefix,
        save_format=image_prefix
    )

filepath = "weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
callbacks_list = [checkpoint]

model1.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=callbacks_list,
        verbose=1
    )
