#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
kmeans聚类测试
"""

import numpy as np
import matplotlib.pyplot as plt
from Unsupervised.Clustering.KMeans import kmeans_utils


def load_data():
    """ 
    加载数据
    """ 
    datList = []
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    return datMat


def clubs_cluster(fileName, imgName, numClust=4):
    """ 
    将Club地址聚类，并绘制到地图上
    :param fileName: 文本数据路径
    :param imgName: 图片路径
    :param numClust: 希望得到的簇数目
    :return:
    """ 

    # 加载数据
    datMat=load_data()

    # biKmeans聚类
    myCentroids, clustAssing = kmeans_utils.biKmeans(datMat, numClust, distMeas=kmeans_utils.distSLC)

    # 绘图
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]

    #绘制地图
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread(imgName)
    ax0.imshow(imgP)

    # 绘制分类结果
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,s=90)
    # 十字标记表示簇中心
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == "__main__":
    fileName = '../../../Data/KMeans/Portland/places.txt'
    imgName = '../../../Data/KMeans/Portland/Portland.png'
    clubs_cluster(fileName=fileName, imgName=imgName, numClust=4)

