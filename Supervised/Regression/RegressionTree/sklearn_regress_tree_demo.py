#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归,CART,Classification And Regression Trees
树回归
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 创建一个随机的数据集，并引入噪声
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    Y = np.sin(X).ravel()
    Y[::5] += 3 * (0.5 - rng.rand(16))

    # 拟合回归模型
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_3 = DecisionTreeRegressor(max_depth=5,min_samples_leaf=6)
    regr_1.fit(X, Y)
    regr_2.fit(X, Y)
    regr_3.fit(X, Y)

    # 预测
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y_1 = regr_1.predict(X_test)
    Y_2 = regr_2.predict(X_test)
    Y_3 = regr_3.predict(X_test)

    # 绘制结果
    plt.figure()
    plt.scatter(X, Y, c="darkorange", label="data")
    plt.plot(X_test, Y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, Y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.plot(X_test, Y_3, color="red", label="max_depth=5,min_samples_leaf=6", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()