#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
Radial Basis Function, RBF, 径向基函数
Radial Basis Function Kernal vs Linear Kernel
"""

import numpy as np
from Supervised.Classification.SupportVectorMachines import svm_utils


def testRbf(k1=1.3):
    # 加载数据
    dataArr, labelArr = svm_utils.load_data('../../../Data/SupportVectorMachines/testSetRBF.txt')

    # 有核SMO训练
    b, alphas = svm_utils.smop(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])

    # 计算w值
    ws = svm_utils.calcWs(alphas, dataArr, labelArr)

    # 画图
    svm_utils.plotfig_SVM(dataArr, labelArr, ws, b, alphas, hasKernal=True)

    # 用训练数据进行预测
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        # 核转换
        kernelEval = svm_utils.kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        # 预测
        # fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    # 载入测试数据进行预测
    dataArr, labelArr = svm_utils.load_data('../../../Data/SupportVectorMachines/testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        # 核转换
        kernelEval = svm_utils.kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        # 预测
        # fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == "__main__":
    # 径向基函数测试
    testRbf(0.8)
