#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
CNN梯度Demo
"""

import numpy as np
from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.CNN import cnn_math_utils

def gradient_check():
    """ 
    梯度检查
    """ 

    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    # 计算forward值
    a, b, cl = cnn_math_utils.load_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)

    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())

    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))


if __name__=='__main__':
    gradient_check()
