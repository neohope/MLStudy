#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CNN前向传播及后向传播Demo
"""

from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.CNN import cnn_math_utils


def test_forward():
    '''
    测试前向传播
    '''
    a, b, cl = cnn_math_utils.load_test()
    cl.forward(a)
    print("cl.output_array:\n", cl.output_array)


def test_backward():
    '''
    测试后向传播
    '''
    a, b, cl = cnn_math_utils.load_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print("cl.filters[0]:\n", cl.filters[0])
    print("cl.filters[1]:\n", cl.filters[1])


if __name__=='__main__':
    test_forward()
    test_backward()
