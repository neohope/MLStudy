#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Max Pooling 卷积分神经网络实现
"""

import numpy as np
from NeuralNetwork.CNN import cnn_math


class MaxPoolingLayer(object):
    """
    CNN卷积分层
    """
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - filter_width) // self.stride + 1
        self.output_height = (input_height -  filter_height) // self.stride + 1
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        """
        前向传播
        """
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (cnn_math.get_patch(input_array[d], i, j, self.filter_width,
                                  self.filter_height, self.stride).max())

    def backward(self, input_array, sensitivity_array):
        """
        后向传播
        """
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = cnn_math.get_patch(input_array[d], i, j, self.filter_width,
                        self.filter_height, self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j * self.stride + l] = sensitivity_array[d, i, j]


def get_max_index(array):
    """ 
    获取一个2D区域的最大值所在的索引
    """ 
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j
