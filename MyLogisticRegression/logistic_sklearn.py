#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归
sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LogisticRegression


def createData():
    """
    构建测试数据
    创建100个点，分为两组
    在X上面，增加一列1
    """
    np.random.seed(6)
    (X,Y) = make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.05,random_state=20)
    X1 = np.c_[np.ones((X.shape[0])),X]
    return X1,Y


if __name__ == '__main__':
    # 创建数据并绘图展示
    X1, Y= createData()
    plt.scatter(X1[:,1],X1[:,2],marker='o',c=Y)
    plt.show()
    plt.scatter(X1[:,1],Y,marker='o',c=Y)
    plt.show()

    # 预测并绘图展示
    clf = LogisticRegression()
    clf.fit(X1, Y)
    predict_y = clf.predict(X1)
    plt.scatter(X1[:,1],X1[:,2],marker='o',c=predict_y)
    plt.show()

    # 输出error和acuracy
    error = sum((predict_y - Y) ** 2)
    print(error)
    accuracy = 1 - (error / 100)
    print(accuracy)
