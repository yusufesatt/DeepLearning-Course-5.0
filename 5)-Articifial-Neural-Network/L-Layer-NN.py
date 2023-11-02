# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:17:33 2023

@author: yusufesat
"""

# %%

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

# Resimlerim x bunun içerisinde
x_l = np.load(r"C:\Users\yusufesat\Desktop\Deep-Learning-5.0\3)-Deep-Learning-ve-Dataset-Tanıtımı\input\Sign-language-digits-dataset\X.npy")
# Resimlerimin classları bunun içinde
Y_l = np.load(r"C:\Users\yusufesat\Desktop\Deep-Learning-5.0\3)-Deep-Learning-ve-Dataset-Tanıtımı\input\Sign-language-digits-dataset\Y.npy")
img_size = 64  # Image size
plt.subplot(1, 2, 1)  # 1 satır 2 sütunluk 1. grafiğim
# x_l içinden 260. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')  # plotumda X,Y eksenlerindeki yazıları kaldırır
plt.subplot(1, 2, 2)  # 1 satır 2 sütunluk 2. grafiğim
# x_l içinden 900. index'e sahip olan resmi alıp 64x64lük resme dönüştürüyor
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')  # plotumda X,Y eksenlerindeki yazıları kaldırır

# %%

# 204den409'a kadar olan resimleri (yani tüm 0'ları) ve 822den1027'ye kadar olan resimleri (1'leri) atar
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)

z = np.zeros(205)  # 205 sıfır resmimizi atıyoruz
o = np.ones(205)  # 205 bir resmimizi atıyoruz
Y = np.concatenate((z, o), axis=0).reshape(
    X.shape[0], 1)  # z ve o değişkenlerimi birleştirir
print("X shape: ", X.shape)  # Output:410, 64, 64 | 64x64lük 410 resmim var
print("Y shape: ", Y.shape)  # Output:410,1

# %%


# Verimin %15ini test için ayırıyorum ve değişkenlerime train,test atıyorum
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)
#
# 1. Veriyi `x` (bağımsız değişkenler) ve `y` (bağımlı değişken) olarak ayırın.
# 2. `train_test_split` fonksiyonu ile veriyi rastgele iki parçaya bölün (genellikle bir eğitim seti ve bir test seti).
# 3. Modeli `x_train` ve `y_train` kullanarak eğitin.
# 4. Modeli `x_test` kullanarak tahminlerde bulunun.
# 5. Tahminleri gerçek değerlerle (`y_test`) karşılaştırarak modelin performansını değerlendirin.
# Output: 348, Train'de kaç resmim var onu öğreniyorum [0] yazarak çok boyutlu dizimin 0. indexini yani resim sayımı alıyorum
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]  # Output 62, **

# %%
# Y datamız bir vektör fakat X datalarımız bir matrix halinde biz model eğitebilmek için X datamızı da vektör haline getireceğiz yani 2D boyutlu hale getireceğiz.
#
# 64x64 piksellik görüntü tek boyutlu şekilde 64 * 64 = 4096 boyutunda bir vektör elde ederim.
X_train_flatten = X_train.reshape(
    number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(
    number_of_test, X_test.shape[1]*X_test.shape[2])

print("X train flatten: ", X_train_flatten.shape)  # Output: (348, 4096)
print("X test flatten: ", X_test_flatten.shape)  # Output: (62, 4096)

# %%
# Matrix çarpımı yapabilmemiz için iki boyutlu diziye geçiş yapıyoruz

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T

print("x train: ", x_train.shape)  # OUT: (4096, 348)
print("x test: ", x_test.shape)  # OUT: (4096, 62)
print("y train: ", y_train.shape)  # OUT: (1, 348)
print("y test: ", y_test.shape)  # OUT: (1, 62)

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
    Z2 = np.dot(parameters["weight2"], A1) + \
        parameters["bias2"]  # weightle a1i çarpıyorum
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
    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    # power() - üssü
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

    grads = {
        "dweight1": dW1,
        "dbias1": db1,
        "dweight2": dW2,
        "dbias2": db2
    }

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


def update_parameters_NN(parameters, grads, learning_rate=0.01):
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
# 2 - Layer Neural Network

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

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)

# %%
# L Layer Neural Network
# Keras kullanacağımız için Transpozunu alıyoruz
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

# %%
# Evaluating the ANN

"""
units: output dimensions of node
kernel_initializer: to initialize weights
activation: activation function, we use relu
input_dim: input dimension that is number of pixels in our images (4096 px)
optimizer: we use adam optimizer
    Adam is one of the most effective optimization algorithms for training neural networks.
    Some advantages of Adam is that relatively low memory requirements and usually works well even with little tuning of hyperparameters
loss: Cost function is same. By the way the name of the cost function is cross-entropy cost function that we use previous parts.

"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    # 4096 input dimension
    # Hidden layer ekledik 8 nodedan oluşan
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    # Hidden layer ekledik 4 nodedan oluşan
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Output layer ekledik 1 nodedan oluşan
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Adam = learning rate'in değişen hali adaptif momentum adıyla
    # Metric = değerlendirmesi accuracy üzerinde
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
# Build ettiğimiz nni çağrıyoruz build_fn ile 
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
# Birden fazla accuracy vererek accuracylerin meanini alıyor cross_Val socre
# 3 kez accuracy bul ve ortalamasını al
mean = accuracies.mean()
variance = accuracies.std()
#Doğruluk varyansı, bir modelin genellemesi üzerindeki güvenilirliğini değerlendirmek için önemli bir ölçüttür. Düşük varyans, modelin farklı veri örnekleri üzerinde tutarlı bir şekilde iyi performans gösterdiği anlamına gelirken, yüksek varyans, modelin farklı veri örneklerine karşı hassas olduğunu ve tutarsız sonuçlar üretebileceğini gösterir.
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))














