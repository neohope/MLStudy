#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归
"""

from numpy import *
from Supervised.Regression.RegressionTree import regress_tree_utils


if __name__ == "__main__":
    #测试拆分数据集
    testMat = mat(eye(4))
    print(testMat)
    mat0, mat1 = regress_tree_utils.binSplitDataSet(testMat, 1, 0.5)
    print(mat0)
    print(mat1)

    # 树回归
    myDat = regress_tree_utils.load_data('../../../Data/RegressionTree/data1.txt')
    myMat = mat(myDat)
    myTree = regress_tree_utils.createTree(myMat)
    print(myTree)

    # 树回归
    myDat = regress_tree_utils.load_data('../../../Data/RegressionTree/data2.txt')
    myMat = mat(myDat)
    myTree = regress_tree_utils.createTree(myMat)
    print(myTree)

