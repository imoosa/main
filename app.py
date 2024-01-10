import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from pandas_datareader import data
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import streamlit as st
from tensorflow.python import tf2
import logging

logging.basicConfig(level=logging.DEBUG)


yf.pdr_override()

def load_data(user_input):
    start = '2010-01-01'
    end = datetime.now()
    df = data.DataReader(user_input, start, end)
    return df

def plot_closing_price(df):
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

def plot_closing_price_with_ma(df):
    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

def plot_closing_price_with_ma200(df):
    st.subheader('Closing Price vs Time Chart with 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

def plot_predictions_vs_original(y_test, y_predicted):
    st.subheader('Predictions vs Original')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original price')
    plt.plot(y_predicted, 'r', label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

def run_streamlit_app():
    st.title('Stock Trend Prediction')
    user_input = st.text_input('Enter Stock Ticker : ', 'AAPL')

    df = load_data(user_input)

    st.subheader('Data Starting from 2010')
    st.write(df.tail())

    plot_closing_price(df)
    plot_closing_price_with_ma(df)
    plot_closing_price_with_ma200(df)

    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    past_100_days = pd.DataFrame(data_training['Close'].tail(100).values, columns=['Close'])
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    plot_predictions_vs_original(y_test, y_predicted)

if __name__ == "__main__":
    run_streamlit_app()
