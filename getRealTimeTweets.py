#!/usr/bin/python3

from __future__ import absolute_import, print_function
from tweepy import OAuthHandler, Stream, StreamListener
import json, os

def readConfig():
    with open('tweets.config') as f:
        lines = f.read().split(os.linesep)
        i = 0
        while i < len(lines):
            l = lines[i]
            i += 1
            l = l.strip()
            sp = l.split(': ')
            if len(sp) == 1:
                continue
            k, v = sp
            k = k.strip()
            if k == 'consumer_key':
                consumer_key = v
            elif k == 'consumer_secret':
                consumer_secret = v
            elif k == 'access_token':
                access_token = v
            elif k == 'access_token_secret':
                access_token_secret = v
            elif k == 'bounding_box':
                closed = False
                if ']' in v:
                    closed = True
                while (not closed):
                    v += lines[i].strip()
                    i += 1
                    if ']' in v:
                        closed = True
                try:
                    locs = eval(v)
                except:
                    print('Error while reading bounding_box')
                    exit()
    return (consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
            locs)

                


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, data):
        tweet = json.loads(data)
        if 'extended_tweet' in tweet:
            txt = str(tweet['extended_tweet']['full_text'])
        elif 'text' in tweet:
            txt = str(tweet['text'])
        else:
            return True
        txt = txt.replace(os.linesep, '')
        print(txt)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':

    consumer_key, consumer_secret, access_token, access_token_secret, locs = readConfig()

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(locations=locs)

