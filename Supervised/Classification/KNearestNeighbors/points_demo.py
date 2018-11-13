#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
利用KNN进行分类处理
判断是哪一类Point
"""

import numpy as np
from Supervised.Classification.KNearestNeighbors import classify_knn


def createDataSet():
    """
    生成测试数据
    根据坐标，将数据划分为两类A和B
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def testPoints():
    """
    给定一个新坐标，判断是哪一类数据
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify_knn.classify0([0.1, 0.1], group, labels, 3))
    print(classify_knn.classify1([0.1, 0.1], group, labels, 3))

if __name__ == '__main__':
    testPoints()
