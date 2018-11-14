#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Max Pooling卷积分神经网络Demo
"""

import numpy as np
from NeuralNetwork.CNN import pool_math


def init_pool_test():
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2], [2, 4]],
         [[3, 5], [8, 2]]], dtype=np.float64)

    mpl = pool_math.MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl


def test_forward():
    """ 
    测试前向传播
    """ 
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a, mpl.output_array))


def test_backward():
    """ 
    测试后向传播
    """ 
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (a, b, mpl.delta_array))


if __name__=='__main__':
    test_forward()
    test_backward()
