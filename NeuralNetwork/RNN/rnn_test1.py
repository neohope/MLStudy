#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
递归神经网络测试
"""

from NeuralNetwork.RNN import rnn
from NeuralNetwork.ANN.activators import RuleActivator


def test():
    """
    前向传播及后向传播
    """
    l = rnn.RecurrentLayer(3, 2, RuleActivator(), 1e-3)
    x, d = rnn.data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d, RuleActivator())
    return l


if __name__ == '__main__':
    test()
