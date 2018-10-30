#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
朴素贝叶斯分类
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


# 高斯朴素贝叶斯
def gaussianNB():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])

    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([[-0.8, -1]]))

    clf_pf = GaussianNB()
    clf_pf.partial_fit(X, Y, np.unique(Y))
    print(clf_pf.predict([[-0.8, -1]]))


# 多项朴素贝叶斯
def multinomialNB():
    X = np.random.randint(5, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 5, 6])
    clf = MultinomialNB()
    clf.fit(X, Y)
    print(clf.predict(X[2:3]))


# 伯努利朴素贝叶斯
def bernoulliNB():
    X = np.random.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    clf = BernoulliNB()
    clf.fit(X, Y)
    print(clf.predict(X[2:3]))


if __name__ == '__main__':
    gaussianNB()
    multinomialNB()
    bernoulliNB()
