#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
递归神经网络测试
"""

from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.RNN import tree_rnn


def test():
    """
    前向传播及后向传播
    """
    children, d = tree_rnn.data_set()
    rnn = tree_rnn.RecursiveLayer(2, 2, IdentityActivator(), 1e-3)
    rnn.forward(children[0], children[1])
    rnn.dump()
    rnn.forward(rnn.root, children[2])
    rnn.dump()
    rnn.backward(d)
    rnn.dump(dump_grad='true')
    return rnn


if __name__ == '__main__':
    test()