#!/usr/bin/env python3

import os
import shutil
import numpy as np

source1 = "./train_dataset"
dest1 = "./validation_dataset"
dest2 = "./test_dataset"
classes = os.listdir(source1)
for c in classes:
    files = os.listdir(source1 + '/' + c)
    if not os.path.exists(dest1 + '/' + c):
        os.makedirs(dest1 + '/' + c)
    if not os.path.exists(dest2 + '/' + c):
        os.makedirs(dest2 + '/' + c)
    for f in files:
        x = np.random.rand(1)
        if x[0] < 0.2:
            shutil.move(
                source1 + '/' + c + '/' + f, dest1 + '/' + c + '/' + f
            )
        elif x[0] < 0.4:
            shutil.move(
                source1 + '/' + c + '/' + f, dest2 + '/' + c + '/' + f
            )
