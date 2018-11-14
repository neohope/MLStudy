#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
LSTM Demo
"""

import numpy as np
from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.LSTM import lstm_math


def gradient_check():
    """ 
    梯度检查
    """ 
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    layer = lstm_math.LstmLayer(3, 2, 1e-3)

    # 计算forward值
    x, d = lstm_math.data_set()
    layer.forward(x[0])
    layer.forward(x[1])

    # 求取sensitivity map
    sensitivity_array = np.ones(layer.h_list[-1].shape,
                                dtype=np.float64)
    # 计算梯度
    layer.backward(x[1], sensitivity_array, IdentityActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(layer.Wfh.shape[0]):
        for j in range(layer.Wfh.shape[1]):
            layer.Wfh[i, j] += epsilon
            layer.reset_state()
            layer.forward(x[0])
            layer.forward(x[1])
            err1 = error_function(layer.h_list[-1])
            layer.Wfh[i, j] -= 2 * epsilon
            layer.reset_state()
            layer.forward(x[0])
            layer.forward(x[1])
            err2 = error_function(layer.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            layer.Wfh[i, j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, layer.Wfh_grad[i, j]))
    return layer


if __name__ == '__main__':
    gradient_check()