#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
梯度下降算法，进行线性回归分析
sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def createData(points):
    """
    创建points符合sin分布的点
    并引入一些噪声
    """
    X=np.linspace(-3,3,points)
    np.random.seed(6)
    Y=np.sin(X)+np.random.uniform(-0.5,0.5,points)
    return X,Y


if __name__ == '__main__':
    # 创建数据并绘图展示
    X, Y = createData(100)
    p=plt.plot(X,Y,'ro')
    plt.axis([-4,4,-2.0,2.0])
    plt.show()

    # 在X前面增加一列1
    X2 = np.c_[np.ones(len(X)), X]

    # 线性回归
    lr = linear_model.LinearRegression()
    lr.fit(X2, Y)

    # 展示拟合结果
    plt.scatter(X, Y, color='red')
    plt.plot(X, lr.predict(X2), color='blue', linewidth=4)
    plt.show()

