#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
LSTM Demo
"""

from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.LSTM import lstm_math


def test():
    """
    前向和后向传播
    """
    x, d = lstm_math.data_set()
    l = lstm_math.LstmLayer(3, 2, 1e-3)
    l.forward(x[0])
    l.forward(x[1])
    l.backward(x[1], d, IdentityActivator())
    return l

if __name__ == '__main__':
    test()