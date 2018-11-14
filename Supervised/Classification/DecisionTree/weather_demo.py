#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
决策树
numpy
"""

from Supervised.Classification.DecisionTree import classify_decision_tree
from Supervised.Classification.DecisionTree import decision_tree_plot


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
    tree = classify_decision_tree.createTree(data, labels)

    # 输出决策树
    print(tree)

    # 可视化展现
    decision_tree_plot.createPlot(tree)
