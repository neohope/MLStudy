#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
CNN math 工具类
"""

import numpy as np
from NeuralNetwork.ANN.activators import IdentityActivator
from NeuralNetwork.CNN import cnn_math


def load_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])

    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])

    cl = cnn_math.ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)

    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)

    cl.filters[0].bias = 1

    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)

    return a, b, cl
