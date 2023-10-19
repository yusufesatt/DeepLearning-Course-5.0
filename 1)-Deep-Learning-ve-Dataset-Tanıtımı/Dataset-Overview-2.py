# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:11:25 2023

@author: yusufesat
"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# %% 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.15, random_state=42) # Verimin %15ini test için ayırıyorum ve değişkenlerime train,test atıyorum
#
# 1. Veriyi `x` (bağımsız değişkenler) ve `y` (bağımlı değişken) olarak ayırın.
# 2. `train_test_split` fonksiyonu ile veriyi rastgele iki parçaya bölün (genellikle bir eğitim seti ve bir test seti).
# 3. Modeli `x_train` ve `y_train` kullanarak eğitin.
# 4. Modeli `x_test` kullanarak tahminlerde bulunun.
# 5. Tahminleri gerçek değerlerle (`y_test`) karşılaştırarak modelin performansını değerlendirin.
number_of_train = X_train.shape[0] # Output: 348, Train'de kaç resmim var onu öğreniyorum [0] yazarak çok boyutlu dizimin 0. indexini yani resim sayımı alıyorum
number_of_test = X_test.shape[0] # Output 62, **

# %%
# Y datamız bir vektör fakat X datalarımız bir matrix halinde biz model eğitebilmek için X datamızı da vektör haline getireceğiz yani 2D boyutlu hale getireceğiz.
# 
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2]) # 64x64 piksellik görüntü tek boyutlu şekilde 64 * 64 = 4096 boyutunda bir vektör elde ederim.
X_test_flatten = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten: ", X_train_flatten.shape) # Output: (348, 4096)
print("X test flatten: ", X_test_flatten.shape)# Output: (62, 4096)

# %%

# Matrix çarpımı yapabilmemiz için iki boyutlu diziye geçiş yapıyoruz

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ",y_test.shape)

# %%



















