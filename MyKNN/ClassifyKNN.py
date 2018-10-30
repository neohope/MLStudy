#!/usr/bin/python
# coding: utf-8

"""
工具类
利用KNN进行分类处理
"""

from numpy import *
from collections import Counter
import operator

def classify0(inX, dataSet, labels, k):
    """
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """

    # 1. 距离计算
    dataSetSize = dataSet.shape[0]

    # tile生成和训练样本对应的矩阵，并与训练样本求差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。
    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    sortedDistIndicies = distances.argsort()

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 3. 排序并返回出现最多的那个类型
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def classify1(inX, dataSet, labels, k):

    """
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """

    """
    1. 计算距离
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """

    dist = sum((inX - dataSet)**2, axis=1)**0.5

    """
    2. k个最近的标签
    函数返回的是索引，因此取前k个索引使用[0 : k]
    将这k个标签存在列表k_labels中
    # """

    k_labels = [labels[index] for index in dist.argsort()[0 : k]]

    """
    3. 出现次数最多的标签即为最终类别
    使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，例如[('lable1', 2)]，因此[0][0]可以取出标签值
    """

    label = Counter(k_labels).most_common(1)[0][0]
    return label