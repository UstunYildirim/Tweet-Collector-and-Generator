# Tweet-Collector-and-Generator

This project has two main functionality. The first one is collecting public tweets using Twitter's API tweepy.

Use getRealTimeTweets.py to collect tweets based on location. Configuration has to be written into tweets.config file. Example is provided at tweets.config.example.

Next, after you collect a large enough amount of tweets, you may feed them to a single layer LSTM to generate new tweets.

For this purpose, you may use 
  - cleanTweets.py to clean up tweets first
  - KerasModel.py to generate a new model
  - KerasTraining.py to train and generate new tweets.

Initial versions of KerasModel.py and KerasTraining.py are based on Keras' example which can be found in https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py . However, the architecture has major differences at the moment.

A pretrained model is stored in model.h5 file.

