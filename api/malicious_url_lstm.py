#Importing Libraries
import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from string import printable 
from keras.utils import pad_sequences
import tensorflow as tf
import json
from tensorflow.keras.models import model_from_json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import regularizers, Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten, Input, ELU, LSTM, Embedding, BatchNormalization, Conv1D, concatenate, MaxPooling1D
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

#Reading data
def read_data():
  df = pd.read_csv("malicious_phish.csv")
  url_int_tokens = [
      [printable.index(x) + 1 for x in url if x in printable] for url in df.iloc[:, 0]
  ]

  max_len = 75
  X = pad_sequences(url_int_tokens, maxlen=max_len)
  le1 = LabelEncoder()

  df['type'] = le1.fit_transform(df['type'])
  target = np.array(df['type'])
  x_train, x_test, target_train, target_test = train_test_split(X, target, test_size=0.25, random_state=42)

  return x_train, x_test, target_train, target_test

x_train, x_test, target_train, target_test = read_data()

#Making the model
max_len = 75
emb_dim = 32
max_vocab_len = 101
lstm_output_size = 32
W_reg = regularizers.l2(1e-4)
epochs_num = 10
batch_size = 32

final_model = Sequential()
final_model.add(Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, embeddings_regularizer=W_reg))
final_model.add(LSTM(lstm_output_size))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_valid, y_valid = x_train[:batch_size], target_train[:batch_size]
x_train2, y_train2 = x_train[batch_size:], target_train[batch_size:]
final_model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs_num, verbose=0)

#Serialize the model and save

final_model.save('lstm.h5')
print("LSTM Model Saved")

#Load the model
lstm = keras.models.load_model('lstm.h5')