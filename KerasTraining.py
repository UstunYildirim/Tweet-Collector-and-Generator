#!/usr/bin/python2

# adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import numpy as np
import random
import sys
import io
import os
from KerasModel import modelFileName, maxlen, getCharSet

# use cleaned tweets
with io.open('Tweets.Cleaned.dat') as f:
    text = f.read()

chars = getCharSet(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# load the model:
model = load_model(modelFileName)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(diversity, seed, numchars=400):
    print('----- diversity:', diversity)

    sentence = seed
    generated = seed
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(numchars):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        preds = preds[-1,:]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

def generate_random_text():
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 0.8, 1.2]:
        generate_text(diversity=diversity,
                seed=text[start_index: start_index + maxlen])
        print()

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    generate_random_text()
    model.save(modelFileName)

if __name__ == '__main__':
    # cut the text in semi-redundant sequences of maxlen characters
    step = 512
    sentences = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen + 1])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen+1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1

    y = x[:,1:,:]
    x = x[:,:-1,:]

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stop = EarlyStopping(monitor='loss',
            min_delta=0,
            patience=1,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
            factor=0.3,
            patience=0,
            min_lr=0.000001)

    model.fit(x, y,
            batch_size=64,
            epochs=60,
            callbacks=[print_callback, early_stop])

    model.save(modelFileName)
