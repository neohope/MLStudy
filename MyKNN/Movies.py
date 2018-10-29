#!/usr/bin/python
# coding: utf-8

from numpy import *
from MyKNN import ClassifyCNN

def createDataSet():
    """
    生成测试数据
    根据坐标，将数据划分为两类A和B
    """
    # [fight,kiss]
    group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ['Love', 'Love', 'Love', 'Action', 'Action', 'Action']
    return group, labels


def testMovies():
    """
    给定一个新坐标，判断是哪一类数据
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(ClassifyCNN.classify0([18, 90], group, labels, 3))
    print(ClassifyCNN.classify1([18, 90], group, labels, 3))

if __name__ == '__main__':
    testMovies()
