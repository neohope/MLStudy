#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
手写识别
"""

import numpy as np
from os import listdir
from Supervised.Classification.SupportVectorMachines import svm_utils


def img2vector(filename):
    """
    读取文件，输出0-1向量
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    """
    枚举文件，每个文件转为0-1向量
    """
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(dataArr, labelArr, dataTest, labelTest, kTup=('rbf', 10)):
    """
    枚举文件，每个文件转为0-1向量
    """

    # 训练
    b, alphas = svm_utils.smop(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])

    # 用训练集判断准确率
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svm_utils.kernelTrans(sVs, datMat[i, :], kTup)
        # 1*m * m*1 = 1*1 单个预测结果
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    # 预测测试数据
    datMat = np.mat(dataTest)
    labelMat = np.mat(labelTest).transpose()
    errorCount = 0
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = svm_utils.kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == "__main__":
    # 导入训练数据
    dataArr, labelArr = loadImages('../../../Data/KNN/handwriting/trainingDigits')
    # 导入测试数据
    dataTest, labelTest = loadImages('../../../Data/KNN/handwriting/testDigits')

    # 训练数据错误率0.055
    # 测试数据错误率0.044
    # 10没有任何作用
    testDigits(dataArr, labelArr, dataTest, labelTest, ('lin', 10))

    # 训练数据错误率0.014
    # 测试数据错误率0.091
    testDigits(dataArr, labelArr, dataTest, labelTest, ('rbf', 50))

    # 训练数据错误率0.000
    # 测试数据错误率0.092
    testDigits(dataArr, labelArr, dataTest, labelTest, ('rbf', 10))

    # 训练数据错误率0.000
    # 测试数据错误率0.004
    testDigits(dataArr, labelArr, dataTest, labelTest, ('rbf', 5))

    # 训练数据错误率0.000
    # 测试数据错误率0.000
    testDigits(dataArr, labelArr, dataTest, labelTest, ('rbf', 0.1))
