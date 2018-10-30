#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
判断是否为鱼的决策树
"""

import copy
from MyDecisionTree import ClassifyDT
from MyDecisionTree import DecisionTreePlot


def createDataSet():
    """
    创建数据集
    """
    # dataSet 前两列是特征，最后一列是分类标签
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # 不浮出水面可以生存，脚蹼
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def fishTest():
    """
    对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    """

    # 创建数据和结果标签
    myDat, labels = createDataSet()

    # 创建决策树
    myTree = ClassifyDT.createTree(myDat, copy.deepcopy(labels))
    print(myTree)

    # 判断[1, 1]是否是鱼
    print(ClassifyDT.classify(myTree, labels, [1, 1]))

    # 可视化展现
    DecisionTreePlot.createPlot(myTree)


if __name__ == "__main__":
    fishTest()
