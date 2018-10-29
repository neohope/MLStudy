#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
利用KNN进行分类处理
判断是哪一类iris
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

"""
这个demo实际上使用了iris数据集
然后对iris数据集进行了分类
"""

# 导入iris数据集
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# 设置图形配色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 用两种方法做了两张图
# 一种是所有数据权重一致
# 一种是距离越远权重越低
for weights in ['uniform', 'distance']:
    # 我们创建了一个knn分类器的实例，并拟合数据
    clf = neighbors.KNeighborsClassifier(3, weights=weights)
    clf.fit(X, Y)

    # 绘制决策边界。为此，我们将为每个分配一个颜色
    # 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # 预测边界中的数据类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 结果绘制
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], 0.02, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"%(3, weights))

plt.show()
