# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:30:54 2023

@author: yusuf
"""
# %%
# Sequence Models

"""
Sequence models plays an over time. 
Speech recognition, natural language process (NLP), music generation
Apples Siri and Google's voice search
Sentiment classification (duygu sınıflandırma) Mesela "bu ders bu dunyadaki en guzel ders" yada "sacma sapan ders cekmissin hocaaa"  
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# %%

dataset_train = pd.read_csv(r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.0/7)-Recurrent-Neural-Network/Dataset/Stock_Price_Train.csv")

dataset_train.head()

# %%

train = dataset_train.loc[:, ["Open"]].values
train

# %%
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train) # Datayı 0-1 arasına scale ettik

print(train_scaled)

# %%
# Creating a data structure with 50 timesteps and 1 output
# 50 tane sample al 1 sonrakini predict et şeklinde ilerleyecek
# 50 tane x_train, 51. y_train olacak
X_train = []
y_train = []

timesteps = 50
for i in range(timesteps, 1258):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# %%
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%
# Create RNN Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout


# Initializing the RNN

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second RNN layer and some dropout regularization
regressor.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN Layer and some Dropout regularization 
regressor.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN Layer and some dropout regularization
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 250, batch_size = 32)

# %%
# Prections and Visualising RNN Model

dataset_test = pd.read_csv(r'C:/Users/yusuf/Desktop/DeepLearning-Course-5.0/7)-Recurrent-Neural-Network/Dataset/Stock_Price_Test.csv')
dataset_test.head()

# %%

real_stock_price = dataset_test.loc[:, ["Open"]].values
print(real_stock_price)

# %%
# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs) # min max scaler
print(inputs)

# %%

# Visualizing

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test) # predict ediyoruz
predicted_stock_price = scaler.inverse_transform(predicted_stock_price) # predict ettiğimiz değerleri 0-1'ken gerçek sayılara çeviriyor

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# epoch = 250 daha güzel sonuç veriyor.


# %%

# long Short Term Memory (LSTM's)

import numpy as np
import math
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# %%

data = pd.read_csv(r'C:/Users/yusuf/Desktop/DeepLearning-Course-5.0/7)-Recurrent-Neural-Network/Dataset/international-airline-passengers.csv', skipfooter=5)
data.head()

# %%

dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.title("international airline passenger")
plt.show()

# %%
# Preprocessing Data

# Reshape

dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")
dataset.shape

# %%
# Scaling

scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(dataset)

# %%

train_size = int(len(dataset) * 0.50)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))

# %%

time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = np.array(dataX)
trainY = np.array(dataY)  

# %%

dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = np.array(dataX)
testY = np.array(dataY)  

# %%

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) # Kerasa uygun şekilde 3 boyutlu matrixe çeviriyoruz
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# %%
# Create LSTM Model

model = Sequential()
model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1)

# %%
#Predictions and Visualising LSTM Model

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# %%
# Visaulising

# shifting train
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
























