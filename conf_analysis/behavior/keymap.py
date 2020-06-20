import numpy as np
import pickle
from . import metadata
import os


mapname = os.path.join(metadata.project, 'key_map.pickle')

try:
    with open(mapname) as f:
        cache = pickle.load(f)
except (IOError, UnicodeDecodeError):
    cache = {'start': 0}


def hash(x):
    x = tuple(x)
    try:
        return cache[x]
    except KeyError:
        cache[x] = max(cache.values()) + 1
        return cache[x]


def save():
    pickle.dump(cache, open(mapname, 'w'), protocol=2)
