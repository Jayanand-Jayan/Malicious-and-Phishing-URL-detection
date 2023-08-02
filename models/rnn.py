import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten, Input, ELU, LSTM, Embedding, BatchNormalization, Convolution1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K

class SimpleLSTM(object):
    def __init__(self) -> None:
        super(SimpleLSTM, self).__init__()
        self.max_len = 75
        self.emb_dim = 32
        self.max_vocab_len = 100
        self.lstm_output_size = 32
        self.W_reg = regularizers.l2(1e-4)

    def build_model(self):
        main_input = Input(shape=(self.max_len,), dtype='int32', name='main_input')

        emb = Embedding(input_dim=self.max_vocab_len, output_dim=self.emb_dim, input_length=self.max_len,
                        embeddings_regularizer=self.W_reg)(main_input)
        
        emb = Dropout(0.2)(emb)

        lstm = LSTM(self.lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)

        output = Dense(1, activation='sigmoid', name='output')(lstm)

        model = Model(inputs=[main_input], outputs=[output])

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy',
                      metrics=['accuracy'])
                      
        return model