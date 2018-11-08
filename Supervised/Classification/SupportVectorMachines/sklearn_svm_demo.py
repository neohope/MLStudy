#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def load_data(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


if __name__ == '__main__':
    # 生成数据
    # np.random.seed(0)
    # X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    # Y = [0] * 20 + [1] * 20

    # 加载数据
    X, Y = load_data('../../../Data/SupportVectorMachines/testSet.txt')
    X = np.mat(X)

    # 拟合一个SVM模型
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    # 获取分割超平面
    w = clf.coef_[0]
    # 斜率
    a = -w[0]/w[1]
    # 采样50个样本
    xx = np.linspace(-2, 10)
    # 二维的直线方程
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # 通过支持向量绘制分割超平面
    print("support_vectors_=", clf.support_vectors_)
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # 图形化展示
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
    plt.scatter(X[:, 0].flat, X[:, 1].flat, c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()
