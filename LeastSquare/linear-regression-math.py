#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
最小二乘法线性回归
math
"""

import numpy as np
import matplotlib.pyplot as plt


def lstsq(X,Y):
    """
    最小二乘法线性回归
    """
    n = len(X)
    sumX,sumY,sumXY,sumXX =0,0,0,0
    for i in range(0,n):
        sumX  += X[i]
        sumY  += Y[i]
        sumXX += X[i]*X[i]
        sumXY += X[i]*Y[i]
    a = (n*sumXY -sumX*sumY)/(n*sumXX -sumX*sumX)
    b = (sumXX*sumY - sumX*sumXY)/(n*sumXX-sumX*sumX)
    return a,b,


if __name__ == '__main__':
    # 数据准备
    X = [1,2,3,4,5,6,7,8,9,10]
    Y = [10,11.5,12,13,14.5,15.5,16.8,17.3,18,18.7]

    #进行拟合
    a,b=lstsq(X,Y)
    print("y = %10.5fx + %10.5f" %(a,b))

    #展示拟合结果
    X1 = np.linspace(0,10)
    Y1 = a * X1 + b
    plt.plot(X1,Y1)
    plt.scatter(X,Y)
    plt.show()

