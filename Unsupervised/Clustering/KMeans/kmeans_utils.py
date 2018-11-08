#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
kmeans工具类
"""

import numpy as np


def load_data(fileName):
    '''
    加载数据集
    '''
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataSet.append(fltLine)
    return dataSet


def distEclud(vecA, vecB):
    '''
    欧氏距离计算
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def distSLC(vecA, vecB):
    '''
    返回地球表面两点间的距离,单位是英里
    给定两个点的经纬度,可以使用球面余弦定理来计算亮点的距离
    :param vecA:
    :param vecB:
    :return:
    '''
    # 经度和维度用角度作为单位,但是sin()和cos()以弧度为输入
    # 可以将江都除以180度然后再诚意圆周率pi转换为弧度
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * \
        np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0


def randCent(dataMat, k):
    '''
    为给定数据集构建一个包含K个随机质心的集合,
    随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
    然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    '''
    # 获取样本数与特征值
    m, n = np.shape(dataMat)

    # 初始化质心
    centroids = np.mat(np.zeros((k, n)))

    # 计算质心
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    '''
    KMeans聚类
    创建K个质心,然后将每个店分配到最近的质心,再重新计算质心。
    这个过程重复数次,直到数据点的簇分配结果不再改变为止
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''

    # 获取样本数和特征数
    m, n = np.shape(dataMat)
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 开始聚类
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False

        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 计算minDist和minIndex
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 簇分配结果发生改变,更新信息
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


def biKmeans(dataMat, k, distMeas=distEclud):
    '''
    二分KMeans聚类
    在给定数据集,簇数和距离计算方法的条件下,计算聚类结果
    SSE: Sum of Sqared Error（误差平方和）
    :param dataMat:
    :param k:
    :param distMeas:
    :return:
    '''
    # 获取样本数和特征数
    m, n = np.shape(dataMat)

    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    centroid0 = np.mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]

    # 遍历数据集中所有点来计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataMat[j, :]) ** 2

    # 对簇不停的进行划分,直到得到想要的簇数目为止
    while (len(centList) < k):
        # 初始化最小SSE为无穷大,用于比较划分前后的SSE
        lowestSSE = np.inf

        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            # 对每一个簇,将该簇中的所有点堪称一个小的数据集
            ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)

            # 如果本次划分的SSE值最小,则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 将选定的数据集通过kmeans函数划分为2个数据集
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 更新最佳质心
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))

        # 更新质心列表,并添加第二个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment
