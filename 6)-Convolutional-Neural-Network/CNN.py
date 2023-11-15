# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:06:43 2023

@author: yusufesat
"""

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# %%
# Dataset preparing

train = pd.read_csv (r'C:\Users\yusuf\Desktop\DeepLearning-Course-5.0\Datasets\Mnist-Data/train.csv')
print(train.shape)
train.head

# %%

test = pd.read_csv(r'C:\Users\yusuf\Desktop\DeepLearning-Course-5.0\Datasets\Mnist-Data/test.csv')
print(test.shape)
test.head

# %%
# Put labels into y_train variable

Y_train = train['label']
# Drop 'label' column
X_train = train.drop(labels = ['label'], axis=1)

# %%
# Visualize number of digits classes 

plt.figure(figsize=(15, 7))
g = sns.countplot(x=Y_train, palette="icefire")  # x=Y_train eklenmiştir
plt.title("Rakam Sınıflarının Sayısı")
plt.show()

print(Y_train.value_counts())
# %%
# Plot some samples

img = X_train.iloc[0].values
img = img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[0,0])
plt.axis('off')
plt.show()

# %%

img = X_train.iloc[3].values
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()

# %%
# Normalization - Reshape - Label Encoding
"""
Normalization:Resimleri 0-1 arasında renklere çevirmek yani siyah beyaz (grayscale) yapmaktır ve cnn'in çalışmasını hızlandırır.

Reshape: Resimlerimiz 28x28 olduğu için keras bunu algılayamaz bu sebeple reshape yaparak 
resimlerimizi 28x28x1 yani 3D Matrix şeklinde oluşturacağız buradaki 1 grayscale anlamına gelir keras 1 kabul eder

Label Encoding: 
    2 => [0,0,1,0,0,0,0,0,0,0]
    4 => [0,0,0,1,0,0,0,0,0,0]
"""

# Normalize the data

X_train = X_train / 255.0
test = test / 255.0
print("X_train shape: ", X_train.shape) # OUT: 
print("test shape: ", test.shape) # OUT:  

# %%
# Reshape

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# %%
# Label encoding

from keras.utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)


# %%
# Train-Test split
# Split the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
print("X_train shape",X_train.shape)
print("X_val shape",X_val.shape)
print("Y_train shape",Y_train.shape)
print("Y_val shape",Y_val.shape)

# %%

# Concolutional Neural Networks (CNN)
# CNN is used for image classification, object detection 

"""
Convolution Layer (relu):
    Kediyi köpekten ayırt eden şeyler: kulak, göz, kuyruk gibi

Pooling Layer:
    Kulak şeklini CL'de çıkardık fakat kulağın tamamına ihtiyacımız yok 
    Bu kulağı ayırt edebileceğim en büyük değeri alırız
    Yani kulağın 5x5 matrixini tutmaktansa bu kulağın maks sayısını tutuyoruz
    Ve bu sayede kulağı tek bir sayıyla ifade ediyoruz
    Bize hız sağlar ve kedi amuda kalkmış bile olsa kulağının tamamına odaklanmaktansa
    Tek bir noktasına odaklanacağımız için ayırt etmeyi kolaylaştırır
    
Flatten: 
    3x3 matrix var ve bunu 9x1'lik vektör haline getiriyoruz ANN kullanıyoruz burada
"""

# %%
# Convoluation Operation

"""
https://i.ibb.co/L02sZ85/gec.jpg
We have some image and feature detector(3*3)
Feature detector does not need to be 3 by 3 matrix. It can be 5 by 5 or 7 by 7.
Feature detector = kernel = filter
Feauture detector detects features like edges or convex shapes. Example, if out input is dog, feature detector can detect features like ear or tail of the dog.
feature map = conv(input image, feature detector). Element wise multiplication of matrices.
feature map = convolved feature
Stride = navigating in input image.
We reduce the size of image. This is important bc code runs faster. However, we lost information.
We create multiple feature maps bc we use multiple feature detectors(filters).
Lets look at gimp. Edge detect: [0,10,0],[10,-4,10],[0,10,0]
"""

# %%
# Same padding

"""
https://ibb.co/fkmNrS2
input size 5x5 lik bir image
etrafına çerçeveyi koyduktan sonra 7x7lik bir image oldu
filtreyi stride=1 ile gezdirdikten sonra tekrar 5x5 lik image oldu
bu sayede veri kaybetmeyi önledik
"""
# %%

# Max pooling
"""
https://ibb.co/BKWLkBj
It makes down-sampling or sub-sampling (Reduces the number of parameters)
It makes the detection of features invariant to scale or orientation changes.
It reduce the amount of parameters and computation in the network, and hence to also control overfitting.
"""

# %%

# Flattening
"""
https://i.ibb.co/3mxD2yr/flattenigng.jpg
Temel olarak bunu yapar 
"""
matrix_2x2 = np.array([[1, 2], [3, 4]])
flattened_array = matrix_2x2.flatten()

# %%

# Fully connected
"""
https://ibb.co/2ZQ35F8
Neurons in a fully connected layer have connections to all activations in the previous layer
Artificial Neural Network
Sigmoid functionu içeren ANN kısmıdır 
"""

# Implementing with keras - Create Model
"""
conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
Dropout: Dropout is a technique where randomly selected neurons are ignored during training
Softmax: multi classlar için kullanılır sigmoide benzer dah genelidir
    Kedi köpek araba kamyon, kedi %70, köpek %40 araba %20 gibi çıktı verir
"""

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# %%

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = "same",
                 activation='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size=(3,3), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
# Fully connected
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

# %%

# Define optimizer
"""
Adam optimizer: Change the learning rate
"""

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)


# %%

# Compile Model
"""
categorical crossentropy
We make binary cross entropy at previous parts and in machine learning tutorial
At this time we use categorical crossentropy. That means that we have multi class.
"""

model.compile(optimizer = optimizer, loss= "categorical_crossentropy", metrics=["accuracy"])

# %%

# Epoch and Batch size

"""
Say you have a dataset of 10 examples (or samples). You have a batch size of 2, and you've specified you want the algorithm to run for 3 epochs. Therefore, in each epoch, you have 5 batches (10/2 = 5). Each batch gets passed through the algorithm, therefore you have 5 iterations per epoch.
reference: https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks
"""

epochs = 10
batch_size = 250

# %%

# Data augmentation 
"""
To avoid overfitting problem, we need to expand artificially our handwritten digit dataset
Alter the training data with small transformations to reproduce the variations of digit.
For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated.
"""

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# %%

# Fit the model 

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size)

# %%

# Evaluate the model 
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%

# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %%






















