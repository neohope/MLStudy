#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归
vs
模型树
vs
线性回归
"""

import numpy as np
from Supervised.Regression.RegressionTree import regress_tree_utils


def load_data(fileName):
    """
    Desc：
        该函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数
    Args:
        fileName 文件名
    Returns:
        dataMat 每一行的数据集array类型
    Raises:
    """
    # 假定最后一列是结果值

    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)
    return dataMat


if __name__ == "__main__":
    # 加载数据
    trainMat = np.mat(load_data('../../../Data/RegressionTree/bikeSpeedVsIq/bikeSpeedVsIq_train.txt'))
    testMat = np.mat(load_data('../../../Data/RegressionTree/bikeSpeedVsIq/bikeSpeedVsIq_test.txt'))

    # 回归树
    myTree1 = regress_tree_utils.createTree(trainMat, ops=(1, 20))
    yHat1 = regress_tree_utils.createForeCast(myTree1, testMat[:, 0])
    #print(myTree1)
    print("回归树:", np.corrcoef(yHat1, testMat[:, 1], rowvar=0)[0, 1])

    # 模型树
    myTree2 = regress_tree_utils.createTree(trainMat, regress_tree_utils.modelLeaf, regress_tree_utils.modelErr, ops=(1, 20))
    yHat2 = regress_tree_utils.createForeCast(myTree2, testMat[:, 0], regress_tree_utils.modelTreeEval)
    #print(myTree2)
    print("模型树:", np.corrcoef(yHat2, testMat[:, 1], rowvar=0)[0, 1])

    # 线性回归
    ws, X, Y = regress_tree_utils.linearSolve(trainMat)
    #print(ws)
    m = len(testMat[:, 0])
    yHat3 = np.mat(np.zeros((m, 1)))
    for i in range(np.shape(testMat)[0]):
        yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print("线性回归:", np.corrcoef(yHat3, testMat[:, 1], rowvar=0)[0, 1])