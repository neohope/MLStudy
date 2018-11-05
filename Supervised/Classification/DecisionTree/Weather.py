#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
决策树
numpy
"""

from Supervised.Classification.DecisionTree import ClassifyDT
from Supervised.Classification.DecisionTree import DecisionTreePlot


def load_data():
    """
    加载数据
    """
    filename = '../../../Data/DecisionTree/whether/whether.csv'
    with open(filename) as fi:
        data = [inst.strip().split(',') for inst in fi.readlines()]
    return data


if __name__ == '__main__':
    # 加载数据并展示
    data=load_data()

    # 数据的Labels
    labels = data[0]
    labels = labels[0:len(labels)-1]
    print(labels)

    data=data[1:]
    print(data)

    # 构造决策树
    tree = ClassifyDT.createTree(data, labels)

    # 输出决策树
    print(tree)

    # 可视化展现
    DecisionTreePlot.createPlot(tree)
