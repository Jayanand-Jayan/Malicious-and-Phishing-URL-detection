# -*- coding: utf-8 -*-
"""Phishing URLs ConvLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sopOQcm48K9jnboJ3puEGL8O_ookAakW
"""

import pandas as pd
import json
import numpy as np
import os
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from string import printable
from keras.utils import pad_sequences
from pathlib import Path
from keras import regularizers, Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten, Input, ELU, LSTM, Embedding, BatchNormalization, Conv1D, concatenate, MaxPooling1D
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import backend as K

def read_data():
  df = pd.read_csv("Phishing_dataset.csv")
  df.drop(df.columns[0], axis=1, inplace=True)
  url_int_tokens = [
      [printable.index(x) + 1 for x in url if x in printable] for url in df.iloc[:, 0]
  ]

  max_len = 75
  X = pad_sequences(url_int_tokens, maxlen=max_len)
  le1 = LabelEncoder()

  df['Label'] = le1.fit_transform(df['Label'])
  target = np.array(df['Label'])
  x_train, x_test, target_train, target_test = train_test_split(X, target, test_size=0.25, random_state=42)

  return x_train, x_test, target_train, target_test

x_train, x_test, target_train, target_test = read_data()

max_len = 75
emb_dim = 32
max_vocab_len = 101
lstm_output_size = 32
W_reg = regularizers.l2(1e-4)
epochs_num = 10
batch_size = 32

final_model = Sequential()
final_model.add(Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, embeddings_regularizer=W_reg))
final_model.add(Conv1D(kernel_size=5, filters=256, padding='same', activation='elu'))
final_model.add(MaxPooling1D(pool_size=4))
final_model.add(Dropout(0.5))
final_model.add(LSTM(lstm_output_size))
final_model.add(Dropout(0.5))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit(x_train, target_train, epochs=epochs_num, batch_size=batch_size)

final_model.save('convlstm.h5')
print("Convolutional LSTM model saved")

convlstm = keras.models.load_model('convlstm.h5')