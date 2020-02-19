# Tweet-Collector-and-Generator

This project has two main functionality. The first one is collecting public tweets using Twitter's API tweepy.

Use getRealTimeTweets.py to collect tweets based on location. Configuration has to be written into tweets.config file. Example is provided at tweets.config.example.

Next, after you collect a large enough amount of tweets, you may feed them to a single layer LSTM to generate new tweets.

For this purpose, you may use KerasExample.py. This file is adapted from Keras' example that can be found in https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py .

A pretrained model is stored in model.h5 file.

