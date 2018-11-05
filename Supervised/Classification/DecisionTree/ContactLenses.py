#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
预测隐形眼镜类型
"""

from Supervised.Classification.DecisionTree import DecisionTreePlot
from Supervised.Classification.DecisionTree import ClassifyDT


def contactLensesTest():
    """
    预测隐形眼镜类型
    """

    # 加载隐形眼镜数据文件
    fr = open('../../../Data/DecisionTree/lenses/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    # 数据的Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    print(lenses)
    print(lensesLabels)

    # 构造预测隐形眼镜的决策树
    lensesTree = ClassifyDT.createTree(lenses, lensesLabels)

    # 输出决策树
    print(lensesTree)

    # 可视化展现
    DecisionTreePlot.createPlot(lensesTree)


if __name__ == "__main__":
    contactLensesTest()
