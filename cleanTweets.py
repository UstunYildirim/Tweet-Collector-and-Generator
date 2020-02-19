#!/usr/bin/python2

import io
import string

# clean raw tweets
printable = set(string.printable)
path = 'Tweets.dat'
with io.open(path, encoding='utf-8') as f:
    text = ''.join(filter(lambda x: x in printable, f.read().lower()))
print('corpus length:', len(text))
with io.open('Tweets.Cleaned.dat', 'w') as f:
    f.write(text)

