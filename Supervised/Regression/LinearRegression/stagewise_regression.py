#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
stagewise regression 分段回归/前向逐步回归
"""

import numpy as np
from Supervised.Regression.LinearRegression import linear_regression_utils


def test():
    # 加载数据
    xArr, yArr = linear_regression_utils.load_data("../../../Data/LinearRegression/abalone/abalone.txt")

    # 拟合
    stageWise(xArr, yArr, 0.01, 200)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = linear_regression_utils.regularize(xMat)
    yM = np.mean(yMat, 0)
    yMat = yMat - yM
    xTx = xMat.T * xMat
    weights = xTx.I * (xMat.T * yMat)
    print(weights.T)


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    stagewise regression 分段回归
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = linear_regression_utils.rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def regularize(xMat):
    """
    规范化
    """
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)  # 计算平均值然后减去它
    inVar = np.var(inMat, 0)  # 计算除以Xi的方差
    inMat = (inMat - inMeans) / inVar
    return inMat


if __name__ == '__main__':
    test()
