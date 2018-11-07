#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归
"""

import matplotlib.pyplot as plt
import numpy as np
from Supervised.Classification.LogisticRegression import classify_logistic


def load_data_set():
    """
    加载数据集
    :return:返回两个数组，普通数组 
        data_arr -- 原始数据的特征
        label_arr -- 原始数据的标签，也就是每条样本对应的类别
    """
    data_arr = []
    label_arr = []
    f = open('../../../Data/LogisticRegression/Set/TestSet.txt', 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        # 增加一列1.0
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr


def plot_best_fit(data_arr, label_arr, weights):
    """
    可视化
    :param weights: 
    :return: 
    """
    data_mat = np.array(data_arr)
    n = np.shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_arr[i]) == 1:
            x_cord1.append(data_mat[i, 1])
            y_cord1.append(data_mat[i, 2])
        else:
            x_cord2.append(data_mat[i, 1])
            y_cord2.append(data_mat[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    #公式y = (-w0 - w1 * x) / w2
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()


if __name__ == '__main__':
    #加载数据
    data_arr, label_arr = load_data_set()

    # 梯度上升法
    weights = classify_logistic.grad_ascent(data_arr, label_arr).getA()
    plot_best_fit(data_arr, label_arr, weights)

    # 随机梯度上升法
    weights = classify_logistic.stoc_grad_ascent(np.array(data_arr), label_arr)
    plot_best_fit(data_arr, label_arr, weights)
