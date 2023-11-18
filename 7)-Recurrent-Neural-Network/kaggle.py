# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:10:31 2023

@author: yusuf
"""

# %% 
# RNN - LSTM
# RNN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.graph_objects as go

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

import warnings

# %%
# Ignore all warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:/Users/yusuf/Desktop/DeepLearning-Course-5.0/7)-Recurrent-Neural-Network/Dataset/ETH-BTC/ETH-BTC-USD.csv')

bitcoin_data = df[df['Currency'] == 'Bitcoin'][['Date', 'Close']].set_index('Date')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin_data)

# Prepare the data for LSTM
def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps):
        a = dataset[i:(i+time_steps), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_steps, 0])
    return np.array(dataX), np.array(dataY)

time_steps = 10  # Adjust as needed
X, y = create_dataset(scaled_data, time_steps)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = y[:train_size], y[train_size:]

# Reshape the input data for LSTM
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, time_steps), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)

# Make predictions on the test set
predictions = model.predict(testX)
predictions = scaler.inverse_transform(predictions)

# Invert the scaling for the actual values
actual_values = scaler.inverse_transform([testY])

# Evaluate the model
mse = mean_squared_error(actual_values[0], predictions[:, 0])
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data.index[train_size+time_steps:], actual_values[0], label='Actual Prices')
plt.plot(bitcoin_data.index[train_size+time_steps:], predictions[:, 0], label='LSTM Predictions', color='red')
plt.title('LSTM Model for Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%

# Separate data for Bitcoin and Ethereum
bitcoin_data = df[df['Currency'] == 'Bitcoin']
ethereum_data = df[df['Currency'] == 'Etherium']  # Adjust the spelling as needed

# Initialize technical indicators
# MACD
bitcoin_data['macd'] = MACD(bitcoin_data['Close']).macd()
ethereum_data['macd'] = MACD(ethereum_data['Close']).macd()

# RSI
bitcoin_data['rsi'] = RSIIndicator(bitcoin_data['Close']).rsi()
ethereum_data['rsi'] = RSIIndicator(ethereum_data['Close']).rsi()

# Bollinger Bands
bb_bands_bitcoin = BollingerBands(bitcoin_data['Close'])
bitcoin_data['bb_upper'] = bb_bands_bitcoin.bollinger_hband()
bitcoin_data['bb_lower'] = bb_bands_bitcoin.bollinger_lband()
bitcoin_data['bb_mid'] = bb_bands_bitcoin.bollinger_mavg()

bb_bands_ethereum = BollingerBands(ethereum_data['Close'])
ethereum_data['bb_upper'] = bb_bands_ethereum.bollinger_hband()
ethereum_data['bb_lower'] = bb_bands_ethereum.bollinger_lband()
ethereum_data['bb_mid'] = bb_bands_ethereum.bollinger_mavg()

# Identify buy and sell signals based on Bollinger Bands
bitcoin_data['Buy_Signal'] = (bitcoin_data['Close'] < bitcoin_data['bb_lower']).astype(int)
bitcoin_data['Sell_Signal'] = (bitcoin_data['Close'] > bitcoin_data['bb_upper']).astype(int)

ethereum_data['Buy_Signal'] = (ethereum_data['Close'] < ethereum_data['bb_lower']).astype(int)
ethereum_data['Sell_Signal'] = (ethereum_data['Close'] > ethereum_data['bb_upper']).astype(int)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Bitcoin prices and technical indicators with signals
plt.subplot(2, 1, 1)
plt.plot(bitcoin_data['Date'], bitcoin_data['Close'], label='Bitcoin Close', color='orange')
plt.plot(bitcoin_data['Date'], bitcoin_data['bb_upper'], label='Upper Bollinger Band', linestyle='--', color='blue')
plt.plot(bitcoin_data['Date'], bitcoin_data['bb_lower'], label='Lower Bollinger Band', linestyle='--', color='blue')
plt.plot(bitcoin_data['Date'], bitcoin_data['bb_mid'], label='Middle Bollinger Band', linestyle='--', color='green')
plt.scatter(bitcoin_data['Date'][bitcoin_data['Buy_Signal'] == 1], bitcoin_data['Close'][bitcoin_data['Buy_Signal'] == 1], label='Buy Signal', marker='^', color='green')
plt.scatter(bitcoin_data['Date'][bitcoin_data['Sell_Signal'] == 1], bitcoin_data['Close'][bitcoin_data['Sell_Signal'] == 1], label='Sell Signal', marker='v', color='red')
plt.title('Bitcoin Prices with Bollinger Bands and Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot Ethereum prices and technical indicators with signals
plt.subplot(2, 1, 2)
plt.plot(ethereum_data['Date'], ethereum_data['Close'], label='Ethereum Close', color='orange')
plt.plot(ethereum_data['Date'], ethereum_data['bb_upper'], label='Upper Bollinger Band', linestyle='--', color='blue')
plt.plot(ethereum_data['Date'], ethereum_data['bb_lower'], label='Lower Bollinger Band', linestyle='--', color='blue')
plt.plot(ethereum_data['Date'], ethereum_data['bb_mid'], label='Middle Bollinger Band', linestyle='--', color='green')
plt.scatter(ethereum_data['Date'][ethereum_data['Buy_Signal'] == 1], ethereum_data['Close'][ethereum_data['Buy_Signal'] == 1], label='Buy Signal', marker='^', color='green')
plt.scatter(ethereum_data['Date'][ethereum_data['Sell_Signal'] == 1], ethereum_data['Close'][ethereum_data['Sell_Signal'] == 1], label='Sell Signal', marker='v', color='red')
plt.title('Ethereum Prices with Bollinger Bands and Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%