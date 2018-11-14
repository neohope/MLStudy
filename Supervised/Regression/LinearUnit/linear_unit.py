#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
线性单元实现
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Supervised.Regression.LinearUnit.perceptron import Perceptron


# 定义激活函数 f
f = lambda x: x


class LinearUnit(Perceptron):
    """
    Desc:
        线性单元类
    Args:
        Perceptron —— 感知器
    Returns:
        None
    """
    def __init__(self, input_num):
        """
        Desc:
            初始化线性单元，设置输入参数的个数
        Args:
            input_num —— 输入参数的个数
        Returns:
            None
        """
        # 初始化感知器类，设置输入参数的个数 input_num 和 激活函数 f
        Perceptron.__init__(self, input_num, f)


def plot(lu, input_vecs, labels):
    """
    Desc:
        将训练好的线性单元对数据的分类情况作图画出来
    Args:
        lu —— 训练好的线性单元
    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(list(map(lambda x: x[0], input_vecs)), labels)
    weights = lu.weights
    bias = lu.bias
    x = range(0, 12, 1)
    y = list(map(lambda x: weights[0] * x + bias, x))
    ax.plot(x, y)
    plt.show()
