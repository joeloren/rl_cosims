import numpy as np


def process_random(random):
    if random is None:
        return np.random.RandomState()
    elif isinstance(random, int):
        return np.random.RandomState(random)
    elif isinstance(random, np.random.RandomState):
        return random
    else:
        print("Some problem with random. Not raising any error!")
        return np.random.RandomState()
