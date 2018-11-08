#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
kmeans聚类
sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


if __name__ == "__main__":
    # 加载数据集
    dataMat = []
    fr = open("../../../Data/KMeans/testSet.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)

    # 训练模型
    km = KMeans(n_clusters=4)

    # 拟合
    km.fit(dataMat)

    # 预测
    km_pred = km.predict(dataMat)

    # 质心
    centers = km.cluster_centers_

    # 可视化展示
    plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
    plt.scatter(centers[:, 1], centers[:, 0], c="r")
    plt.show()
