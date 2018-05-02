#!/usr/bin/env python3

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

os.environ['KERAS_BACKEND'] = 'tensorflow'

test_data_dir = "test_dataset"
img_width = 150
img_height = 150
aug_test_data_dir = "aug_test_data_dir"
test_prefix = "test"
image_prefix = "jpeg"
train_set_mean = np.array([132.59479491, 127.57392624, 131.1389212])
train_set_std = np.array([57.95601941, 65.03537769, 67.78074134])

if not os.path.exists(aug_test_data_dir):
    os.makedirs(aug_test_data_dir)

test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=60,
        zoom_range=0.2
    )

test_datagen.mean = train_set_mean
test_datagen.std = train_set_std

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode="categorical",
        save_to_dir=aug_test_data_dir,
        save_prefix=test_prefix,
        save_format=image_prefix
    )

print(test_generator.class_indices)

model = load_model('model.hdf5')
metrics_values = model.evaluate_generator(test_generator)

print(model.metrics_names)
print(metrics_values)
