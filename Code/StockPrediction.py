# -*- coding: utf-8 -*-
"""
Isha Agrawal
https://github.com/isha-git/Stock-Data-Prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error
from prettytable import PrettyTable

from sklearn.metrics import mean_squared_error

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Importing training data from Google Drive


# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data

downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('Train Dataset.csv')  
dataset_train = pd.read_csv('Train Dataset.csv')

"""Selecting six features for training and forecasting- Open Price, High Price, Low price, Last Price, Close Price, Average Price"""

training_set = dataset_train.iloc[:, 4:10].values

"""Normalization: To scale the training dataset Scikit-Learn’s MinMaxScaler with range zero to one is used."""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = training_set_scaled[:-1, :]
y_train = training_set_scaled[1:, :]

X_train = np.reshape(np.asarray(X_train), (1, np.asarray(X_train).shape[0], np.asarray(X_train).shape[1]))
y_train = np.reshape(np.asarray(y_train), (1, np.asarray(y_train).shape[0], np.asarray(y_train).shape[1]))

"""Creating the LSTM Model and importing the required libraries"""

from tensorflow.python.keras import Sequential
from tensorflow import keras
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Conv2D, MaxPooling2D, Flatten,TimeDistributed
from keras.optimizers import Adam, SGD

"""Adam Optimizer is used and the loss function used here is Mean Sqaured error."""

model = Sequential()

model.add(LSTM(units=64, return_sequences = True,  input_shape=(None, np.asarray(X_train).shape[2]),  name="LSTM1"))
model.add(LSTM(units=64, return_sequences = True, name="LSTM2"))
model.add(LSTM(units=32, return_sequences = True, name="LSTM3"))
model.add(LSTM(units=32, return_sequences = True, name="LSTM4"))
model.add(LSTM(units=16, name="LSTM5"))
model.add(Dense(units=6, name="Dense1"))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss= 'mean_squared_error')
hist = model.fit(X_train,y_train,epochs=100,batch_size=32)

model.summary()

plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Traning Loss')
plt.xlabel('Epochs')
plt.show()

"""Function to predict parameters of Day 1, Day 2 and Day 3 using the above trained model"""

def print_output(dataset_test):

  # Selecting six features to be forecasted
  real_stock_price = dataset_test.iloc[:, 4:10].values

  sc = MinMaxScaler(feature_range=(0,1))
  testing_set_scaled = sc.fit_transform(real_stock_price)

  X_test = testing_set_scaled[:-1, :]
  y_test = testing_set_scaled[1:, :]

  X_test = np.reshape(X_test, (1, np.asarray(X_test).shape[0], np.asarray(X_test).shape[1]))
  y_test = np.reshape(y_test, (1, np.asarray(y_test).shape[0], np.asarray(y_test).shape[1]))

  final_output=[]
  prediction_error=[]


  # The rows correspond to dates on which the parameters will be predicted
  y_1 = y_test[0:1, -4:-3, :]
  x = X_test[0:1, -4:-3, :]
  day_1 = model.predict(x)

  inp = np.reshape(day_1, (1, day_1.shape[0], day_1.shape[1]))
  y_2 = y_test[0:1, -3:-2, :]
  x = X_test[0:1, -3:-2, :]
  day_2 = model.predict(x)

  inp = np.reshape(day_2, (1, day_2.shape[0], day_2.shape[1]))
  y_3 = y_test[0:1, -2:-1, :]
  x = X_test[0:1, -2:-1, :]
  day_3 = model.predict(x)


  # Perfoming inverse transform on output to restore the range
  day_1 = sc.inverse_transform(day_1)
  y_1 = sc.inverse_transform(y_1[0, 0:1, :])

  day_2 = sc.inverse_transform(day_2)
  y_2 = sc.inverse_transform(y_2[0, 0:1, :])

  day_3 = sc.inverse_transform(day_3)
  y_3 = sc.inverse_transform(y_3[0, 0:1, :])


  # Formatting data to 2 decimal places
  for i in range(0, 6):
    day_1[0, i] = round(day_1[0, i], 2)
    day_2[0, i] = round(day_2[0, i], 2)
    day_3[0, i] = round(day_3[0, i], 2)

    y_1[0, i] = round(y_1[0, i], 2)
    y_2[0, i] = round(y_2[0, i], 2)
    y_3[0, i] = round(y_3[0, i], 2)


  # Calculating average loss of the six parameters predicted by the model
  mean_open = abs(round((day_1[0, 0] + day_2[0, 0] + day_3[0, 0] - y_1[0, 0] - y_2[0, 0] - y_3[0, 0])/3, 2))
  mean_high = abs(round((day_1[0, 1] + day_2[0, 1] + day_3[0, 1] - y_1[0, 1] - y_2[0, 1] - y_3[0, 1])/3, 2))
  mean_low = abs(round((day_1[0, 2] + day_2[0, 2] + day_3[0, 2] - y_1[0, 2] - y_2[0, 2] - y_3[0, 2])/3, 2))
  mean_last = abs(round((day_1[0, 3] + day_2[0, 3] + day_3[0, 3] - y_1[0, 3] - y_2[0, 3] - y_3[0, 3])/3, 2))
  mean_close = abs(round((day_1[0, 4] + day_2[0, 4] + day_3[0, 4] - y_1[0, 4] - y_2[0, 4] - y_3[0, 4])/3, 2))
  mean_average = abs(round((day_1[0, 5] + day_2[0, 5] + day_3[0, 5] - y_1[0, 5] - y_2[0, 5] - y_3[0, 5])/3, 2))  
  

  # Representing data in tabular form using PrettyTable
  t = PrettyTable()
  t.field_names = ["Data", "Date", "Open Price", "High Price", "Low Price", "Last Price", "Close Price", "Average Price", "Average Loss"]

  t.add_row(["Ground Truth", dataset_test.iloc[-4, 2], y_1[0, 0], y_1[0, 1], y_1[0, 2], y_1[0, 3], y_1[0, 4], y_1[0, 5], round(mean_absolute_error(y_1[0, :], np.asarray(day_1)[0, :]), 2)])
  t.add_row(["Predicted", "", day_1[0, 0], day_1[0, 1], day_1[0, 2], day_1[0, 3], day_1[0, 4], day_1[0, 5], ""])
  t.add_row(["", "", "", "", "", "", "", "", ""])

  t.add_row(["Ground Truth", dataset_test.iloc[-3, 2], y_2[0, 0], y_2[0, 1], y_2[0, 2], y_2[0, 3], y_2[0, 4], y_2[0, 5], round(mean_absolute_error(y_2[0, :], np.asarray(day_2)[0, :]), 2)])
  t.add_row(["Predicted", "", day_2[0, 0], day_2[0, 1], day_2[0, 2], day_2[0, 3], day_2[0, 4], day_2[0, 5], ""])
  t.add_row(["", "", "", "", "", "", "", "", ""])

  t.add_row(["Ground Truth", dataset_test.iloc[-2, 2], y_3[0, 0], y_3[0, 1], y_3[0, 2], y_3[0, 3], y_3[0, 4], y_3[0, 5], round(mean_absolute_error(y_3[0, :], np.asarray(day_3)[0, :]), 2)])
  t.add_row(["Predicted", "", day_3[0, 0], day_3[0, 1], day_3[0, 2], day_3[0, 3], day_3[0, 4], day_3[0, 5], ""])


  # Calculating average loss for the company
  loss_ = 0
  loss_ = loss_ + mean_absolute_error(day_1, y_1)
  loss_ = loss_ + mean_absolute_error(day_2, y_2)
  loss_ = loss_ + mean_absolute_error(day_3, y_3)

  t.add_row(["", "", "", "", "", "", "", "", ""])
  t.add_row(["Average Loss", "", mean_open, mean_high, mean_low, mean_last, mean_close, mean_average, round(loss_/3, 2)])

  print(t)

"""Importing test data from Google Drive.

Companies used for testing- Raymond, Indian Oil Corporation, Bharat Electronics, SAIL, Spice Jet
"""

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data
downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('Raymond.csv')  
dataset_test = pd.read_csv('Raymond.csv')

print("RAYMOND")
print_output(dataset_test)
print("\n")

#########################################################################

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data
downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('Indian Oil Corporation.csv')  
dataset_test = pd.read_csv('Indian Oil Corporation.csv')

print("Indian Oil Corporation")
print_output(dataset_test)
print("\n")

#########################################################################

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data
downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('Bharat Electronics.csv')  
dataset_test = pd.read_csv('Bharat Electronics.csv')

print("BHARAT ELECTRONICS")
print_output(dataset_test)
print("\n")

#########################################################################

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data
downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('SAIL.csv')  
dataset_test = pd.read_csv('SAIL.csv')

print("SAIL")
print_output(dataset_test)
print("\n")

#########################################################################

link = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"				# Replace this with the shareable link of your data
downloaded = drive.CreateFile({'id':link}) 
downloaded.GetContentFile('SPICEJET.csv')  
dataset_test = pd.read_csv('SPICEJET.csv')

print("Spice Jet")
print_output(dataset_test)
print("\n")

"""The “Open” column represents the opening price for shares that day and the “High” column represents the highest price shares reached that day. “Low” represents the lowest share price for the day, “Last” represents the price at which the last transaction for a share went through. “Close” represents the price shares ended at for the day.

**REFERENCES**

1. https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
2. https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
"""