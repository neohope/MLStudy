#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
预测鲍鱼年龄
Locally weighted linear regression 局部加权线性回归
对比
最小二乘法线性回归
"""

import numpy as np
from Supervised.Regression.LinearRegression import linear_regression_utils


def test():
    """
    Desc:
        预测鲍鱼的年龄
    Args:
        None
    Returns:
        None
    """
    # 加载数据
    abX, abY = linear_regression_utils.load_data("../../../Data/LinearRegression/abalone/abalone.txt")

    # 使用不同的核进行预测
    oldyHat01 = linear_regression_utils.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    oldyHat1 = linear_regression_utils.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    oldyHat10 = linear_regression_utils.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", linear_regression_utils.rssError(abY[0:99], oldyHat01.T))
    print("old yHat1 error Size is :", linear_regression_utils.rssError(abY[0:99], oldyHat1.T))
    print("old yHat10 error Size is :", linear_regression_utils.rssError(abY[0:99], oldyHat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = linear_regression_utils.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("new yHat01 error Size is :", linear_regression_utils.rssError(abY[0:99], newyHat01.T))
    newyHat1 = linear_regression_utils.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("new yHat1 error Size is :", linear_regression_utils.rssError(abY[0:99], newyHat1.T))
    newyHat10 = linear_regression_utils.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("new yHat10 error Size is :", linear_regression_utils.rssError(abY[0:99], newyHat10.T))

    # 线性回归，与上面的计算进行比较
    xMat = np.mat(abX[0:99])
    yMat = np.mat(abY[0:99]).T
    xTx = np.mat(abX[0:99]).T * np.mat(abX[0:99])
    standWs =  xTx.I * (xMat.T * yMat)
    standyHat = np.mat(abX[100:199]) * standWs
    print("stand regress error Size is:", linear_regression_utils.rssError(abY[100:199], standyHat.T.A))


if __name__ == '__main__':
    test()
