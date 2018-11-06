#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归,CART,Classification And Regression Trees
vs
AdaBoost回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


def test():
    # 创建数据集，并引入噪声
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    Y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

    # 拟合
    regr_1 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
    regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
    regr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=5),n_estimators=300, random_state=rng)

    regr_1.fit(X, Y)
    regr_2.fit(X, Y)
    regr_3.fit(X, Y)

    # 预测
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)

    # Plot the results
    plt.figure()
    plt.scatter(X, Y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="max_depth=2,n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="max_depth=5,n_estimators=1", linewidth=2)
    plt.plot(X, y_3, c="b", label="max_depth=5,n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
