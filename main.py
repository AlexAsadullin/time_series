import tensorflow as tf
import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


def get_data():
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    df = df[5::6]
    df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    temp = df['T (degC)']
    temp.plot()
    return df, temp


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


def build_model():
    model1 = Sequential()
    model1.add(InputLayer((5, 1)))
    model1.add(LSTM(64))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1, 'linear'))
    print(model1.summary())
    return model1


def train_test_model(model, X_train1, y_train1, X_val1, y_val1):
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])
    model.save('stock_model')
    train_predictions = model.predict(X_train1).flatten()
    train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
    return model, train_results, train_predictions


if __name__ == '__main__':
    df, temp = get_data()
    X1, y1 = df_to_X_y(temp, window_size=5)
    X_train1, y_train1 = X1[:60000], y1[:60000]
    
    X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
    X_test1, y_test1 = X1[65000:], y1[65000:]
    print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)
    
    model1 = build_model()
    model, train_results, train_predictions = train_test_model(model=model1,
                                                               X_train1=X_train1,
                                                               y_train1=y_train1
                                                               X_val1=X_val1
                                                               y_val1=y_val1)
    
    plt.plot(train_results['Train Predictions'][50:100])
    plt.plot(train_results['Actuals'][50:100])
    
    val_predictions = model1.predict(X_val1).flatten()
    val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
    plt.plot(val_results['Val Predictions'][:100])
    plt.plot(val_results['Actuals'][:100])
