# Ivy Vasquez Sandoval
"""
This is representative of independent research into Neural Networks
exploring how these models are created, initialized, used, etc.
"""

# Verify GPU memory growth
import tensorflow as tf
config = tf.config.experimental.list_physical_devices('GPU')
for device in config:
    tf.config.experimental.set_memory_growth(device, True)

# Data Preprocessing
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the training dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = np.asarray(dataset_train.iloc[:, 1:2])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the Recurrent Neural Network
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

# Initializing the RNN
RNN_Model = Sequential()

# Adding the LSTM layers with Dropout
RNN_Model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
RNN_Model.add(Dropout(0.2))
RNN_Model.add(LSTM(units = 50, return_sequences = True))
RNN_Model.add(Dropout(0.2))
RNN_Model.add(LSTM(units = 50, return_sequences = True))
RNN_Model.add(Dropout(0.2))
RNN_Model.add(LSTM(units = 50))
RNN_Model.add(Dropout(0.2))

# Adding the output layer
RNN_Model.add(Dense(units = 1))

# Compiling the RNN
RNN_Model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the traning set
RNN_Model.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Import the test dataset
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = np.asarray(dataset_test.iloc[:, 1:2])

# Generate prediction dataset
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = np.asarray(dataset_total[len(dataset_total) - len(dataset_test) - 60:])
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_set = RNN_Model.predict(X_test)
pred_set = sc.inverse_transform(pred_set)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(pred_set, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
