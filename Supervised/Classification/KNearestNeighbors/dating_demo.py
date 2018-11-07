#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
利用KNN进行分类处理
判断约会对象是否合适
"""

import matplotlib.pyplot as plt
from numpy import *

from Supervised.Classification.KNearestNeighbors import classify_knn


def file2matrix(filename):
    """
    从文件导入训练数据
    返回数据矩阵returnMat和对应的类别classLabelVector
    """

    # 获得文件中的数据行的行数
    # 生成对应的空矩阵
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm0(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet-tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def autoNorm1(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、极差
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet, ranges, minVals

def draw(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

def datingClassTest():
    """
    对约会网站进行数据测试
    """

    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('../../../Data/KNN/dating/dating.txt')
    draw(datingDataMat, datingLabels)

    # 归一化数据
    normMat, ranges, minVals = autoNorm0(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]

    # 测试数据集范围,训练数据集比例=1-hoRatio
    hoRatio = 0.1
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify_knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = 10
    ffMiles = 10000
    iceCream = 0.5
    datingDataMat, datingLabels = file2matrix('../../../Data/KNN/dating/dating.txt')
    normMat, ranges, minVals = autoNorm0(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify_knn.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

if __name__ == '__main__':
    #datingClassTest()
    classifyPerson()
