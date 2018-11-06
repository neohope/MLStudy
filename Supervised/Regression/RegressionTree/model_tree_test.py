#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归
"""
import numpy as np
from Supervised.Regression.RegressionTree import regress_tree_utils


if __name__ == "__main__":
    # 模型树
    myDat = regress_tree_utils.load_data('../../../Data/RegressionTree/data4.txt')
    myMat = np.mat(myDat)
    myTree = regress_tree_utils.createTree(myMat, regress_tree_utils.modelLeaf, regress_tree_utils.modelErr)
    print(myTree)

