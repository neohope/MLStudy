#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
knn算法进行分类处理
numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets.samples_generator import make_blobs


def createData():
    """
    创建50个点，分为两组
    """
    np.random.seed(6)
    (X, Y) = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.95, random_state=50)
    return X,Y


def get_eculidean_distance(point,k):
    """
    计算欧几里德距离，返回最近的k个点
    """
    euc_distance = np.sqrt(np.sum((X - point)**2 , axis=1))
    return np.argsort(euc_distance)[0:k]


def predict(prediction_points, k, Y):
    """
    通过KNN分组
    """
    points_labels = []
    for point in prediction_points:
        distances = get_eculidean_distance(point, k)
        results = []
        for index in distances:
            results.append(Y[index])
        label = Counter(results).most_common(1)
        points_labels.append([point, label[0][0]])

    return points_labels


def get_accuracy(predictions,Y):
    """
    计算精度
    """
    error=np.sum((predictions-Y)**2)
    accuracy=100-(error/len(Y))*100
    return accuracy


if __name__ == '__main__':
    # 创建数据并绘图展示
    X, Y=createData()
    plt.scatter(X[:,0],X[:,1],marker='o',c=Y)
    plt.show()

    #用不同的K值，查找最有效的K数值
    acc=[]
    for k in range(1,10):
        results=predict(X,k,Y)
        predictions=[]
        for result in results:
            predictions.append(result[1])
        acc.append([get_accuracy(predictions,Y),k])

    plotx = []
    ploty = []
    for a in acc:
        plotx.append(a[1])
        ploty.append(a[0])

    plt.plot(plotx, ploty)
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.show()

    # 创建测试数据并展示
    prediction_points=[[-2,-4],[-3,-6],[1,0],[6,4],[-6,4]]
    prediction_points=np.array(prediction_points)
    plt.scatter(X[:,0],X[:,1],marker='o',c=Y)
    plt.scatter(prediction_points[:,0],prediction_points[:,1],marker='o')
    plt.show()

    # KNN处理测试数据
    results=predict(prediction_points,3,Y)
    for result in results:
        print("Point = ",result[0])
        print("Class = ",result[1])
        print()
