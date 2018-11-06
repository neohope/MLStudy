#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Locally weighted linear regression 局部加权线性回归
"""

import numpy as np
import matplotlib.pylab as plt
from Supervised.Regression.LinearRegression import linear_regression_utils


def test():
    # 加载数据
    xArr, yArr = linear_regression_utils.load_data("../../../Data/LinearRegression/data.txt")

    #yHat = lr_utils.lwlrTest(xArr, xArr, yArr, 1)
    yHat = linear_regression_utils.lwlrTest(xArr, xArr, yArr, 0.01)
    #yHat = lr_utils.lwlrTest(xArr, xArr, yArr, 0.003)

    # 重新排序
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort = xMat[srtInd][:, 0, :]

    # 图形展示
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter([xMat[:, 1].flatten().A[0]], [np.mat(yArr).T.flatten().A[0]], s=2, c='red')
    plt.show()


if __name__ == '__main__':
    test()