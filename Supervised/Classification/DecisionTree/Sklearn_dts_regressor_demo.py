#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
利用决策树进行数据拟合
"""

# 引入必要的模型和库
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 创建一个随机的数据集作为X
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
# Sin（X）作为Y并引入一些噪声
Y = np.sin(X).ravel()
Y[::5] += 3 * (0.5 - rng.rand(16))

# 拟合回归模型
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=3)
regr_3 = DecisionTreeRegressor(max_depth=4)
regr_4 = DecisionTreeRegressor(max_depth=5)
regr_5 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=6)

regr_1.fit(X, Y)
regr_2.fit(X, Y)
regr_3.fit(X, Y)
regr_4.fit(X, Y)
regr_5.fit(X, Y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)
y_4 = regr_3.predict(X_test)
y_5 = regr_3.predict(X_test)

# 绘制结果
plt.figure()
plt.scatter(X, Y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="red", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="green", label="max_depth=3", linewidth=2)
plt.plot(X_test, y_3, color="yellow", label="max_depth=4", linewidth=2)
plt.plot(X_test, y_4, color="blue", label="max_depth=5", linewidth=2)
plt.plot(X_test, y_5, color="black", label="max_depth=5, min_samples_leaf=6", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
