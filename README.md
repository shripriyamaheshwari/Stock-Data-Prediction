# Problem Statement
To predict various stock market parameters (Open Price, High Price, Low Price, Last Price, Close Price, Average Price) using the LSTM/GRU model and predict Day1-Day2-Day3 forecast of 5 more companies and evaluate its performance.

# LSTM vs GRU
While Gated Recurrent Units (GRU) use lesser training paramaters (therefore, lesser memory), and are faster than Long Short-Term Memory Networks (LSTM), LSTM networks have a higher accuracy. As for this project, accuracy is more important than run-time, LSTM networks have been used for forecasting.

# Libraries
1. Numpy <br>
2. Matplotlib <br>
3. Pandas <br>
4. Sklearn <br>
5. Keras <br>
6. Tensorflow <br>

Environment- Python 3.7.3, Windows 10 <br>
The code was developed on Google Colab.

# Dataset
<a href="https://www1.nseindia.com/products/content/equities/equities/eq_security.htm">National Stock Exchange India </a> was referred to obtain company-wise data of 12 Fortune 500 India companies. <br>
The companies used for training are-

1. Bombay Dyeing and Manufacturing <br>
2. Bosch <br>
3. Britannia <br>
4. Coal India <br>
5. Hindustan Construction Company <br>
6. Hindustan Unilever <br>
7. Mahindra and Mahindra Financial Services <br>
8. Nestle <br>
9. Oracle Financial Services Software <br>
10. Reliance Industries <br>
11. State Bank of India <br>
12. Whirlpool of India <br>

<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/Dataset.PNG" width=1500>
Data of past three years (27 March 2017 - 24 March 2020) was used for training, which is approxmately 700 data samples for each company.

# Architecture
The network contains 5 LSTM layers, and a dense layer in the end containing 6 units (to predict 6 parameters). Adam optimizer Mean Squared Error have been used for training for 100 epochs.

<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/ModelArchitecture.PNG" width = 500>

# Results
The model was tested on 5 Fortune 500 India companies-
1. Raymond
2. Indian Oil Corporation
3. Bharat Electronics
4. Steel Authority of India Limited (SAIL)
5. Spice Jet

Mean Absolute Error (L1 error) has been used for quantitative evaluation of the results.

<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/Raymond.PNG" width = 1000>
<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/BharatElectronics.PNG" width = 1000>
<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/IndianOilCorporation.PNG" width = 1000>
<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/SAIL.PNG" width = 1000>
<img src = "https://github.com/isha-git/Stock-Data-Prediction/blob/master/images/SpiceJet.PNG" width = 1000>

# Discussions
It can be observed that the model gives moderately high error. This could be because of the different variations in the stock prices of the companies with respect to time. Accordingly, the model can be improved for better performance.
