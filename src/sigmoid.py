import numpy as np

def sigmoid(z):
#SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.

    g = 1/(1 + np.exp(-z))
    return g