#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
无核函数示例
"""

import numpy as np
from Supervised.Classification.SupportVectorMachines import svm_utils


if __name__ == "__main__":
    # 获取特征和目标变量
    dataArr, labelArr = svm_utils.load_data('../../../Data/SupportVectorMachines/testSet.txt')

    # 聚类分析
    # b是常量值， alphas是拉格朗日乘子
    b, alphas = svm_utils.smop(dataArr, labelArr, 0.6, 0.001, 40)

    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])

    # 计算w值
    ws = svm_utils.calcWs(alphas, dataArr, labelArr)

    # 画图
    svm_utils.plotfig_SVM(dataArr, labelArr, ws, b, alphas)
