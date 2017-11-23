import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hyper_tang(x):
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))

def linear(x):
    return x
