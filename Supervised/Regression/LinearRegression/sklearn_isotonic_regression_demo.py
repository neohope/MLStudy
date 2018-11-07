#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
保序回归 isotonic regression
对比
线性回归
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state


if __name__ == '__main__':
    # 生成样本数据并增加噪音
    n = 100
    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(-50, 50, size=(n,)) + 50 * np.log(1 + np.arange(n))

    # 保序回归进行拟合
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y)

    # 线性回归的进行拟合
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)

    # 图形化展示
    segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(0.5 * np.ones(n))

    fig = plt.figure()
    plt.plot(x, y, 'r.', markersize=12)
    plt.plot(x, y_, 'g.-', markersize=12)
    plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    plt.gca().add_collection(lc)
    plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
    plt.title('Isotonic regression')
    plt.show()
