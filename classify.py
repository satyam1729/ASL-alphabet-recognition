#!/usr/bin/env python3

import os
import pathlib
from string import ascii_lowercase

folder_name = '/dataset/'
pwd = os.getcwd()+folder_name
directories = os.listdir(pwd)
for c in ascii_lowercase:
    pathlib.Path(pwd+c).mkdir(parents=True, exist_ok=True)
for i in range(10):
    pathlib.Path(pwd+str(i)).mkdir(parents=True, exist_ok=True)
for directory in directories:
    files = os.listdir(pwd+directory)
    for file in files:
        os.rename(pwd+directory+'/'+file, pwd+file[6]+'/'+file)
    os.rmdir(pwd+directory)
