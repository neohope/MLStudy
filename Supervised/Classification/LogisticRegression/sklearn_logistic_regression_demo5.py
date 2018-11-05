#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归与线性回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def Sigmoid(x):
    # 阶跃函数
    return 1 / (1 + np.exp(-x))


def test():
    # 创建数据集并引入噪声
    n_samples = 100
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    Y = (X > 0).astype(np.float)
    X[X>0] *= 4
    X += .3 * np.random.normal(size=n_samples)
    X = X[:, np.newaxis]

    # 运行分类器
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, Y)

    # 图形化展示
    plt.figure(1, figsize=(4, 3))
    plt.clf()

    #绘制数据点
    plt.scatter(X.ravel(), Y, color='black', zorder=20)

    # 逻辑回归曲线
    X_test = np.linspace(-5, 10, 300)
    loss = Sigmoid(X_test * clf.coef_ + clf.intercept_).ravel()
    plt.plot(X_test, loss, color='red', linewidth=3)

    # 绘制线性回归曲线
    ols = linear_model.LinearRegression()
    ols.fit(X, Y)
    plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
    plt.axhline(.5, color='.5')

    plt.ylabel('y')
    plt.xlabel('X')
    plt.xticks(range(-5, 10))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-.25, 1.25)
    plt.xlim(-4, 10)
    plt.legend(('Logistic Regression Model', 'Linear Regression Model'), loc="lower right", fontsize='small')
    plt.show()

if __name__ == '__main__':
    test()
