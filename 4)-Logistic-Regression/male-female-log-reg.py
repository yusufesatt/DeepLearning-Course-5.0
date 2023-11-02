# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:06:15 2023

@author: yusufesat
"""

# %%

from sklearn.model_selection import train_test_split
from sklearn import linear_model
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%

train_path = r"C:/Users/yusufesat/Desktop/Deep-Learning-5.0/4)-Logistic-Regression/dataset1"
classes_id = []
target = []
for idx, classes in enumerate(os.listdir(train_path)):
    file_names = os.listdir(train_path+'/'+classes)
    images = np.zeros((len(file_names), 64, 64))
    for (i, im_path) in tqdm(enumerate(file_names)):
        img = cv2.imread(train_path+'/'+classes + '/'+im_path, 0)
        img = cv2.resize(img, (64, 64))
        images[i, :] = img
        target.append(idx)
    classes_id.append(images)


x = np.concatenate((classes_id), axis=0)
y = np.array(target)

# %%
# Reshape yapmama gerek yok çünkü ilk cellde 64,64 yapıyoruz.
plt.subplot(1, 2, 1)  # 1 satır 2 sütunluk 1. grafiğim
# x_l içinden 260. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.imshow(x[4999])
plt.axis('off')  # plotumda X,Y eksenlerindeki yazıları kaldırır
plt.subplot(1, 2, 2)  # 1 satır 2 sütunluk 2. grafiğim
# x_l içinden 900. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.imshow(x[5000])
plt.axis('off')  # plotumda X,Y eksenlerindeki yazıları kaldırır

# %%
z = np.zeros(5000)
o = np.ones(5000)
Y = np.concatenate((z, o), axis=0).reshape(
    x.shape[0], 1)  # z ve o değişkenlerimi birleştirir
print("X shape: ", x.shape)
print("Y shape: ", Y.shape)  # Output:410,1

# %%


X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=42)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

# %%

# 64x64 piksellik görüntü tek boyutlu şekilde 64 * 64 = 4096 boyutunda bir vektör elde ederim.
X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])

print("X train flatten: ", X_train_flatten.shape)  # Output: (348, 4096)
print("X test flatten: ", X_test_flatten.shape)

# %%

x_train = X_train_flatten.T/255
x_test = X_test_flatten.T/255
y_train = Y_train.T
y_test = Y_test.T


print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)

# %%


def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


w, b = initialize_weights_and_bias(4096)

# %%


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
# %%


def forward_propagation(w, b, x_train, y_train):
    # weight ve x_train çarpımından sonra bias ile toplamım sonucu z değerim ortaya çıkar.
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)  # 0-1 arası değerim
    print(y_head)
    print(y_train)

    # loss = -y_train * np.log(y_head + 1e-5) - (1 - y_train) * np.log(1 - y_head + 1e-5)
    # BUNU GÖSTER

    loss = -y_train*np.log(y_head)-(1-y_train) * np.log(1-y_head)  # Loss function
    print(loss)
    # Gerçek etiketler (y_train) ve model tahminlerinin (y_head) logaritmasıyla hesaplanır.
    # İkinci ve dördüncü terimler, doğru sınıf tahminlerinde logaritmaları sıfır yapar
    cost = (np.sum(loss))/x_train.shape[1]  # Cost function
    # Hesaplanan log kaybı değerlerinin ortalamasını alarak toplam kaybı hesaplar.
    # Daha sonra bu değer, veri noktalarının sayısına (x_train.shape[1]) bölünür.
    # Böylece, her veri noktası için ortalama kaybı temsil eden bir değeri elde edersiniz.
    return cost


# %%

def forward_backward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}

    return cost, gradients

# %%


def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))

    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


parameters, gradients, cost_list = update(
    w, b, x_train, y_train, learning_rate=0.01, number_of_iteration=150)

# %%


def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)

    Y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, 1] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    return Y_prediction

# %%


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_of_iteration):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, number_of_iteration)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, x_test, y_test,learning_rate=0.01, number_of_iteration=800)

# %%

logreg = linear_model.LogisticRegression(random_state=42, max_iter=150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
