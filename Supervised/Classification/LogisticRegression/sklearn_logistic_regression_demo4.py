#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归进行分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model


def test():
    # 加载数据，数据分为3类
    # 采用样本数据的前两个feature
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target
    h = .02  # 网格中的步长
    logreg = linear_model.LogisticRegression(C=1e5)

    # 拟合数据
    logreg.fit(X, Y)

    # 计算决策边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # 绘制预测结果
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # 绘制训练点
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == '__main__':
    test()

