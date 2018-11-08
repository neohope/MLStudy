#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
kmeans聚类测试
"""

import numpy as np
import matplotlib.pyplot as plt
from Unsupervised.Clustering.KMeans import kmeans_utils


if __name__ == "__main__":
    # 加载数据
    dataMat = np.mat(kmeans_utils.load_data('../../../Data/KMeans/testSet.txt'))

    # 训练4中心
    centroids, clusterAssment = kmeans_utils.kMeans(dataMat, 4)

    # 可视化展示
    plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=np.array(clusterAssment)[:, 0])
    plt.scatter(np.array(centroids)[:, 1], np.array(centroids)[:, 0], c="r")
    plt.show()

    # 训练3中心
    centroids, clusterAssment = kmeans_utils.kMeans(dataMat, 3)

    # 可视化展示
    plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=np.array(clusterAssment)[:, 0])
    plt.scatter(np.array(centroids)[:, 1], np.array(centroids)[:, 0], c="r")
    plt.show()

    # 训练2中心
    centroids, clusterAssment = kmeans_utils.kMeans(dataMat, 2)

    # 可视化展示
    plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=np.array(clusterAssment)[:, 0])
    plt.scatter(np.array(centroids)[:, 1], np.array(centroids)[:, 0], c="r")
    plt.show()
