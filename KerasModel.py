#!/usr/bin/python2

# adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import io

modelFileName = 'model.h5'

maxlen = 10 # time steps for the LSTM

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
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.save(modelFileName)

