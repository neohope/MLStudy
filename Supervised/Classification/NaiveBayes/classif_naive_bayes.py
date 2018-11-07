#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
工具类
利用决策树进行分类处理

贝叶斯公式
p(xy)=p(x|y)p(y)=p(y|x)p(x)
p(x|y)=p(y|x)p(x)/p(y)
"""

import numpy as np


def train_naive_bayes(train_mat, train_category):
    """
    朴素贝叶斯分类
    :param train_mat: 训练文本
    :param train_category: 对应的文本类别
    :return:
    """

    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])

    # 侮辱性文件的出现概率
    pos_abusive = np.sum(train_category) / train_doc_num

    # 单词出现的次数
    p0num = np.ones(words_num)
    p1num = np.ones(words_num)

    # 整个数据集单词出现的次数
    p0num_all = 2.0
    p1num_all = 2.0

    # 遍历所有的文件，分别计算此文件中出现的单词出现的频率
    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])

    p1vec = np.log(p1num / p1num_all)
    p0vec = np.log(p0num / p0num_all)
    return p0vec, p1vec, pos_abusive


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):
    """
    朴素贝叶斯分类
    :param vec2classify: 待分类的向量[0,1,1,1,1...]
    :param p0vec: 类别0向量，[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]
    :param p1vec: 类别1向量，[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]
    :param p_class1: 判断类别1出现概率
    :return: 类别1 or 0
    """

    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class1)

    if p1 > p0:
        return 1
    else:
        return 0
