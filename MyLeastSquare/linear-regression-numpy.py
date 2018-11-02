#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
最小二乘法线性回归
numpy
"""

import numpy as np
import matplotlib.pyplot as plt


def lstsq(X,Y):
    """
    最小二乘法线性回归
    """
    A = np.vstack([X, np.ones(len(X))]).T
    return np.linalg.lstsq(A, Y)[0]


if __name__ == '__main__':
    # 数据准备
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y = [10, 11.5, 12, 13, 14.5, 15.5, 16.8, 17.3, 18, 18.7]

    #进行拟合
    a, b = lstsq(X,Y)
    print("y = %10.5fx + %10.5f" % (a, b))

    #展示拟合结果
    X1 = np.array(X)
    Y1 = np.array(Y)
    plt.plot(X1,Y1,'o',label='data',markersize=10)
    plt.plot(X1,a*X1+b,'b',label='line')
    plt.show()
