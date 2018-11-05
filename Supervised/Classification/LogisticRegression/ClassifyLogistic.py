#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归
"""

import numpy as np


def sigmoid(x):
    """
    阶跃函数
    """
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_arr, class_labels):
    """
    梯度上升法，其实就是因为使用了极大似然估计
    :param data_arr: 传入的就是一个普通的数组，当然你传入一个二维的ndarray也行
    :param class_labels: class_labels 是类别标签，它是一个 1*100 的行向量
    :return:
    """

    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()# 变成矩阵之后进行转置

    # m->数据量，样本数 n->特征数
    m, n = np.shape(data_mat)

    # learning rate
    alpha = 0.001
    # 最大迭代次数
    max_cycles = 500
    # 初始化回归系数
    weights = np.ones((n, 1))

    for k in range(max_cycles):
        # 这里是点乘  m x 3 dot 3 x 1
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def stoc_grad_ascent(data_mat, class_labels, num_iter=100):
    """
    随机梯度上升，使用随机的一个样本来更新回归系数，并进行迭代
    :param data_mat: 输入数据的数据特征（除去最后一列）,ndarray
    :param class_labels: 输入数据的类别标签（最后一列数据
    :param num_iter: 迭代次数
    :return: 得到的最佳回归系数
    """
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机产生一个 0～len()之间的一个值
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights


def classify_vector(in_x, weights):
    """
    最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    :param in_x: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return:
    """
    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0

