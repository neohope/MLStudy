#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
递归神经网络测试
"""

import numpy as np
from NeuralNetwork.RNN import rnn
from NeuralNetwork.ANN.activators import IdentityActivator


def gradient_check():
    """ 
    梯度检查
    """ 
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    rl = rnn.RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x, d = rnn.data_set()
    rl.forward(x[0])
    rl.forward(x[1])

    # 求取sensitivity map
    sensitivity_array = np.ones(rl.state_list[-1].shape, dtype=np.float64)

    # 计算梯度
    rl.backward(sensitivity_array, IdentityActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i, j] += epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err1 = error_function(rl.state_list[-1])
            rl.W[i, j] -= 2 * epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err2 = error_function(rl.state_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            rl.W[i, j] += epsilon
            print('weights(%d,%d): expected - actural %f - %f' % (i, j, expect_grad, rl.gradient[i, j]))


if __name__ == '__main__':
    gradient_check()
