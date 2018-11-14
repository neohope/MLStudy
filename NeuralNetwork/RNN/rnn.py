#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
递归神经网络实现
"""

import numpy as np
from functools import reduce


class RecurrentLayer(object):
    def __init__(self, input_width, state_width, activator, learning_rate):
        """
        递归神经网络
        """
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width, 1)))  # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))  # 初始化W

    def forward(self, input_array):
        """ 
        前向传播
        """ 
        self.times += 1
        state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def backward(self, sensitivity_array, activator):
        """ 
        BPTT后向传播
        """ 
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def update(self):
        """ 
        按照梯度下降，更新权重
        """ 
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []  # 用来保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width, 1)))
        self.delta_list.append(sensitivity_array)

        # 迭代计算每个时刻的误差项
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self, k, activator):
        """ 
        根据k+1时刻的delta计算k时刻的delta
        """ 
        state = self.state_list[k + 1].copy()
        element_wise_op(self.state_list[k + 1],
                        activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k + 1].T, self.W),
            np.diag(state[:, 0])).T

    def calc_gradient(self):
        self.gradient_list = []  # 保存各个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros(
                (self.state_width, self.state_width)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)

        # 实际的梯度是各个时刻梯度之和
        self.gradient = reduce(
            lambda a, b: a + b, self.gradient_list,
            self.gradient_list[0])  # [0]被初始化为0且没有被修改过

    def calc_gradient_t(self, t):
        """ 
        计算每个时刻t权重的梯度
        """ 
        gradient = np.dot(self.delta_list[t], self.state_list[t - 1].T)
        self.gradient_list[t] = gradient

    def reset_state(self):
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((self.state_width, 1)))  # 初始化s0


def element_wise_op(array, op):
    """ 
    对numpy数组进行element wise操作
    """ 
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def data_set():
    x = [np.array([[1], [2], [3]]), np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d
