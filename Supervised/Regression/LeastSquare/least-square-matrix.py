#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
最小二乘法线性回归
matrix
"""

import numpy as np
import matplotlib.pylab as plt
from Supervised.Regression.LinearRegression import utils


def stand_least_square(xMat, yMat):
    '''
    Description：
        线性回归
    Args:
        X ：输入的样本数据，包含每个样本数据的 feature
        Y ：对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws：回归系数
    '''

    # 矩阵转置
    yMat = yMat.T

    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat

    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I * (xMat.T * yMat)
    return ws


if __name__ == '__main__':
    # 数据准备
    X, Y = utils.loadDataSet("../../../Data/Regression/data.txt")
    xMat = np.mat(X)
    yMat = np.mat(Y)

    # 拟合
    ws = stand_least_square(xMat, yMat)

    # 图形展示
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()