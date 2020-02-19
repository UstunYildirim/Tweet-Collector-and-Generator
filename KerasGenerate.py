#!/usr/bin/python2

from KerasTraining import generate_text
from KerasModel import maxlen
import sys
import warnings

warnings.filterwarnings('ignore')
print("Note: Warnings are ignored.")

try:
    diversity = float(sys.argv[1])
    seed = sys.argv[2]
    if len(seed) != maxlen:
        raise Exception()
except:
    print('Please provide a diversity value and a seed of length {}.'.format(maxlen))
    exit()

generate_text(diversity=float(sys.argv[1]), seed=sys.argv[2], numchars=1000)
print()

