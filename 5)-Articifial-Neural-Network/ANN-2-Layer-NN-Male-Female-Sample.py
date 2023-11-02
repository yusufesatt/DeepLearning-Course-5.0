# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:39:46 2023

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
# Initialize parameters

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    # 3 node'dan oluşan 2 katmanlı bir neural network
    parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,  # 3 node'un x_trainin feature'larına karşılık gelen ağırlıklar.
                  # 1. katmandaki bias değerleri başlangıçta 0 atanır
                  "bias1": np.zeros((3, 1)),
                  # 2. katmandaki nodeların output özelliklerine karşılık gelir.
                  "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0], 1))}
    return parameters


# %%
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# %%
# Forward propagation
# Piksel ile weightleri çarpıp bias ile toplayıp Z elde ediyoruz
# Ve z değerini tanh fonksiyonuna sokuyoruz


def forward_propagation_NN(x_train, parameters):
    # weightle piksellerimi çarpıyorum
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)  # tanh'a yukarda bulduğum değeri buluyorum
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"] # weightle a1i çarpıyorum
    A2 = sigmoid(Z2)  # sigmoide sokup a2 değerimi alıyorum yani y_head

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


# %%
# Loss and Cost Function
"""
Args:
    A2: NN modelinin tahminleri
    Y: Gerçek output değerler (etiketler)

Return:
    Cost 
"""


def compute_cost_NN(A2, Y, parameters):
    # Tahminlerle gerçek değerler arasındaki farkı hesapla (loss fonksiyonu olarak log-likelihood kullanılıyor)
    logprobs = np.multiply(np.log(A2), Y)
    # Toplam maliyeti hesapla (negatif log-likelihood)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost

# %%
# Backward propagation
# weight ve bias'ların türevlerini hesaplar


def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


# %%
# update parameters
"""
    weight ve bias değerlerini günceller.
    
    Args:
    parameters -- ağırlık ve bias değerlerini içeren sözlük
    grads -- ağırlık ve bias türevlerini içeren sözlük
    learning_rate -- öğrenme oranı, varsayılan değer 0.01
    
    Returns:
    parameters -- güncellenmiş ağırlık ve bias değerlerini içeren sözlük
"""


def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters

# %%
# Prediction


def predict_NN(parameters, x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test, parameters)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

# %%
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=4500)
