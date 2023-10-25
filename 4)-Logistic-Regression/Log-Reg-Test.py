# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:39:41 2023

@author: alooo
"""

# %%

import os
import numpy as np
import cv2
from tqdm import tqdm 
train_path = r"dataset/training_set/training_set"
classes_id = []
target=[]
for idx,classes in enumerate(os.listdir(train_path)):
    file_names=os.listdir(train_path+'/'+classes)
    images=np.zeros((len(file_names),64,64))
    for( i,im_path) in tqdm(enumerate(file_names)):
        img=cv2.imread(train_path+'/'+classes +'/'+im_path,0)
        img=cv2.resize(img, (64,64))
        images[i,:]=img
        target.append(idx)
    classes_id.append(images)


print(np.concatenate((classes_id),axis=0).shape,np.array(target).shape)

# %%


