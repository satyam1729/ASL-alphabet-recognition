#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:50:42 2018

@author: satyam
"""

import os
import pathlib
pwd=os.getcwd()+'/dataset/'
directories=os.listdir(pwd)
from string import ascii_lowercase
for c in ascii_lowercase:
    pathlib.Path(pwd+c).mkdir(parents=True, exist_ok=True)
for i in range(10):
    pathlib.Path(pwd+str(i)).mkdir(parents=True, exist_ok=True)
    

for directory in directories:
    files=os.listdir(pwd+directory)
    for file in files:
        os.rename(pwd+directory+'/'+file,pwd+file[6]+'/'+file)
    os.rmdir(pwd+directory)
