#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ridge regression 岭回归/脊回归/吉洪诺夫正则化
"""

import matplotlib.pylab as plt
from Supervised.Regression.LinearRegression import linear_regression_utils


def test():
    abX, abY = linear_regression_utils.load_data("../../../Data/LinearRegression/abalone/abalone.txt")
    ridgeWeights = linear_regression_utils.ridgeTest(abX, abY)
    print(ridgeWeights.shape)

    # 查看系数变化
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ridgeWeights)
    plt.show()


if __name__ == '__main__':
    test()
