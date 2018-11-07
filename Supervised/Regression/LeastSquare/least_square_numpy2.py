#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
最小二乘法线性回归
numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


# 目标函数
def prepareData():
    # 随机选择10个点作为X
    X = np.linspace(0, 1, 10)
    Y = np.sin(2 * np.pi * X)
    Y1 = [np.random.normal(0, 0.1) + y for y in Y]
    return  X,Y1


# 多项式函数
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差函数
def residuals_func(p, y, x):
    # 只有这一句会过拟合
    ret = fit_func(p, x) - y

    # 增加这两句会欠拟合
    regularization = 0.001
    ret = np.append(ret, np.sqrt(regularization) * p)
    return ret


if __name__ == '__main__':
    # 数据准备
    X, Y1=prepareData()

    # 利用9次多项式进行拟合，随机初始化多项式参数，并进行拟合
    n = 9
    p_init = np.random.randn(n)
    plsq = leastsq(residuals_func, p_init, args=(Y1, X))

    # 展示拟合结果
    x_points = np.linspace(0, 1, 1000)
    plt.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
    plt.plot(X, Y1, 'bo', label='with noise')
    plt.legend()
    plt.show()

