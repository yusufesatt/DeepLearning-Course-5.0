# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:47:03 2023

@author: yusufesat
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from subprocess import check_output

# %%
x_l = np.load(r"C:\Users\yusufesat\Desktop\Deep-Learning-5.0\1)-Deep-Learning-ve-Dataset-Tanıtımı\input\Sign-language-digits-dataset\X.npy") # Resimlerim x bunun içerisinde
Y_l = np.load(r"C:\Users\yusufesat\Desktop\Deep-Learning-5.0\1)-Deep-Learning-ve-Dataset-Tanıtımı\input\Sign-language-digits-dataset\Y.npy") # Resimlerimin classları bunun içinde
img_size = 64 # Image size
plt.subplot(1, 2, 1) # 1 satır 2 sütunluk 1. grafiğim
plt.imshow(x_l[260].reshape(img_size, img_size)) # x_l içinden 260. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.axis('off') # plotumda X,Y eksenlerindeki yazıları kaldırır
plt.subplot(1, 2, 2) # 1 satır 2 sütunluk 2. grafiğim
plt.imshow(x_l[900].reshape(img_size, img_size)) # x_l içinden 900. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.axis('off') # plotumda X,Y eksenlerindeki yazıları kaldırır

# %%

X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0) # 204den409'a kadar olan resimleri (yani tüm 0'ları) ve 822den1027'ye kadar olan resimleri (1'leri) atar

z = np.zeros(205) # 205 sıfır resmimizi atıyoruz
o = np.ones(205) # 205 bir resmimizi atıyoruz
Y = np.concatenate((z,o), axis = 0).reshape(X.shape[0],1) # z ve o değişkenlerimi birleştirir 
print("X shape: ", X.shape) # Output:410, 64, 64 | 64x64lük 410 resmim var 
print("Y shape: ", Y.shape) # Output:410,1