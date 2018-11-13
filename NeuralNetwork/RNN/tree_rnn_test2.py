#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
递归神经网络测试
"""

import numpy as np
from NeuralNetwork.RNN import tree_rnn
from NeuralNetwork.ANN.activators import IdentityActivator


def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    rnn = tree_rnn.RecursiveLayer(2, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x, d = tree_rnn.data_set()
    rnn.forward(x[0], x[1])
    rnn.forward(rnn.root, x[2])

    # 求取sensitivity map
    sensitivity_array = np.ones((rnn.node_width, 1),
                                dtype=np.float64)
    # 计算梯度
    rnn.backward(sensitivity_array)

    # 检查梯度
    epsilon = 10e-4
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            rnn.W[i, j] += epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err1 = error_function(rnn.root.data)
            rnn.W[i, j] -= 2 * epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err2 = error_function(rnn.root.data)
            expect_grad = (err1 - err2) / (2 * epsilon)
            rnn.W[i, j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, rnn.W_grad[i, j]))
    return rnn


if __name__ == '__main__':
    gradient_check()