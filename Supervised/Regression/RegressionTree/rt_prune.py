#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归
剪枝/分支合并
"""

import numpy as np
from Supervised.Regression.RegressionTree import regress_tree_utils


def prune(tree, testData):
    """
    Desc:
        从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
    Args:
        tree -- 待剪枝的树
        testData -- 剪枝所需要的测试数据 testData 
    Returns:
        tree -- 剪枝完成的树
    """

    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
    if np.shape(testData)[0] == 0:
        return getMean(tree)

    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if (regress_tree_utils.isTree(tree['right']) or regress_tree_utils.isTree(tree['left'])):
        lSet, rSet = regress_tree_utils.binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if regress_tree_utils.isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)

    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if regress_tree_utils.isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not regress_tree_utils.isTree(tree['left']) and not regress_tree_utils.isTree(tree['right']):
        lSet, rSet = regress_tree_utils.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


def getMean(tree):
    """
    Desc:
        计算左右枝丫的均值
        从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
        对 tree 进行递归处理，即返回树平均值。
    Args:
        tree -- 输入的树
    Returns:
        返回 tree 节点的平均值
    """
    if regress_tree_utils.isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if regress_tree_utils.isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


if __name__ == "__main__":
    # 预剪枝就是：提前设置最大误差数和最少元素数
    myDat = regress_tree_utils.load_data('../../../Data/RegressionTree/data3.txt')
    myMat = np.mat(myDat)
    myTree = regress_tree_utils.createTree(myMat, ops=(0, 1))
    print(myTree)

    # 后剪枝就是：通过测试数据，对预测模型进行合并判断
    myDatTest = regress_tree_utils.load_data('../../../Data/RegressionTree/data3test.txt')
    myMat2Test = np.mat(myDatTest)
    myFinalTree = prune(myTree, myMat2Test)
    print(myFinalTree)
