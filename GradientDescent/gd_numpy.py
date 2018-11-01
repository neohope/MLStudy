#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
梯度下降算法，进行线性回归分析
numpy
"""

import numpy as np
import matplotlib.pyplot as plt


def createData(points):
    """
    创建points个符合sin分布的点
    并引入一些噪声
    """
    X=np.linspace(-3,3,points)
    np.random.seed(6)
    Y=np.sin(X)+np.random.uniform(-0.5,0.5,points)
    return X,Y


def gradient_descent(X2, Y, lrate, epochs, W):
    """
    梯度下降
    """
    total_expected_error = 0
    errorlist = []
    finalepoch = 0

    for i in range(epochs):
        # 计算误差
        predictedY = X2.dot(W)
        error = (predictedY - Y) ** 2
        total_error = np.sum(error)

        # d/dϴ =error*x.T
        # 梯度下降
        gradient = X2.T.dot(error) / X2.shape[0]

        # 每一百次记录一次错误情况
        if i % 100 == 0:
            errorlist.append(total_error)
            finalepoch += 1

        # 相邻两次训练没有明显提升，则结束训练
        if np.abs(total_expected_error - total_error) < 0.0005:
            return errorlist, finalepoch
        total_expected_error = total_error

        # 进行系数修正
        W += -lrate * gradient

    return errorlist, finalepoch


if __name__ == '__main__':
    # 创建数据并绘图展示
    X, Y = createData(100)
    p=plt.plot(X,Y,'ro')
    plt.axis([-4,4,-2.0,2.0])
    plt.show()

    # 在X前面增加一列1
    X2 = np.c_[np.ones(len(X)), X]

    # 初始化系数
    W = np.random.uniform(size=X2.shape[1], )

    # 训练5000次
    total_error, finalepoch = gradient_descent(X2, Y, 0.001, 5000, W)

    # 展示错误下降情况
    plt.plot(range(finalepoch), total_error)
    plt.xlabel("epochs in 100's")
    plt.ylabel("error")
    plt.show()

    #展示拟合结果
    plt.plot(X, Y, 'ro')
    plt.plot(X, X2.dot(W), 'b')
    plt.show()
