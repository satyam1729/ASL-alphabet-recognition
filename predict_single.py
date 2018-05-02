#!/usr/bin/env python3

import os
import cv2
import shutil
import numpy as np
from keras.models import load_model
from scipy.ndimage import rotate


os.environ['KERAS_BACKEND'] = 'tensorflow'

test_data_dir = "test_dataset"
img_width = 150
img_height = 150
model = load_model('model.hdf5')

classes = os.listdir(test_data_dir)
for c in classes:
    predictions = {}
    files = os.listdir('{0}/{1}'.format(test_data_dir, c))
    for f in files:
        path = './{0}/{1}/{2}'.format(test_data_dir, c, f)
        img1 = cv2.imread(path)
        img1 = cv2.resize(img1, (img_height, img_width))
        img1 = np.expand_dims(img1, axis=0)
        x = model.predict_classes(img1)[0]
        if x not in predictions:
            predictions[x] = 1;
        else:
            predictions[x] = predictions[x] + 1
    print('For {0} predictions are {1}'.format(c, predictions))

img = cv2.imread('./check_model/1.jpg')
img = cv2.resize(img, (img_width, img_height))
img = rotate(img, 270)
img = np.expand_dims(img, axis=0)
x = model.predict_classes(img)
print(x)
