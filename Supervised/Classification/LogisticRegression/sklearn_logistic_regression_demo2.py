#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归路径
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import l1_min_c


def test():
    # 加载数据
    # 3类数据去掉第2类
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    X = X[Y != 2]
    Y = Y[Y != 2]
    # 减去均值，让相对差距更明显
    X -= np.mean(X, 0)

    #创建数据空间
    cs = l1_min_c(X, Y, loss='log') * np.logspace(0, 3)

    # 拟合数据
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

    # 拟合路径
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, Y)
        coefs_.append(clf.coef_.ravel().copy())
    coefs_ = np.array(coefs_)
    #print(coefs_)

    # 绘制路径
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    test()
