#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf

source1 = "./train_dataset"
classes = os.listdir(source1)
train_set_mean = np.array([132.59479491, 127.57392624, 131.1389212])
std = np.array([0, 0, 0])
counter = 0
for c in classes:
    files = os.listdir(source1 + '/' + c)
    print('******** Currently processing {0} *********'.format(c))
    for f in files:
        path = '{0}/{1}/{2}'.format(source1, c, f)
        pil_img = tf.keras.preprocessing.image.load_img(
                path,
                grayscale=False,
                target_size=(150, 150)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(
            pil_img,
            data_format=None
        )
        subract_square = np.square(img_array - train_set_mean)
        subtract_square_mean = np.mean(subract_square, axis=(0, 1))
        std = np.add(std, subtract_square_mean)
        counter = counter + 1
        if not counter % 199:
            print('************')
            print('{0} image'.format(counter))
            print(subtract_square_mean)
            print(std)
            print('************')

std = np.sqrt(np.divide(std, counter))

print(train_set_mean, std)
