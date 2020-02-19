#!/usr/bin/python2

# adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, LSTM, LocallyConnected1D, Permute
from keras.optimizers import Adam
import io

modelFileName = 'model.h5'

maxlen = 12 # time steps for the LSTM

def getCharSet(text):
    return sorted(list(set(text)))

if __name__ == '__main__':
    # use cleaned tweets
    with io.open('Tweets.Cleaned.dat') as f:
        text = f.read()

    chars = getCharSet(text)
    print('total chars:', len(chars))

    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    units = maxlen*16
    model.add(LSTM(units, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Permute((2,1)))
    model.add(LocallyConnected1D(len(chars), (units//maxlen), strides=(units//maxlen)-1, activation='softmax'))

    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    model.save(modelFileName)

