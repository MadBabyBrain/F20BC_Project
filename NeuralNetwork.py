import math
import numpy as np


def sigmoid(val):
    return 1 / 1 + math.exp(-val)


def relu(val):
    return max(0, val)


def tangent(val):
    return math.tanh(val)


def sigmoid_derivative(val):
    return (1 - sigmoid(val)) * sigmoid(val)


def relu_derivative(val):
    out = 0
    if val > 0:
        out = 1
    return out


def tangent_derivative(val):
    return 1 - (math.tanh(val) ** 2)


class NeuralNetwork:
    def __init__(self, layers):
        pass
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
def main():
    n = NeuralNetwork(layers=[3, 3, 3])
    pass

if __name__ == "__main__":
    main()
