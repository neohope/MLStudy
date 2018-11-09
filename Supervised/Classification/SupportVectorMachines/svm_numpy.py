#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def load_data():
    """
    构建测试数据
    50个数据，分为两组
    在X上面，增加一列1
    """
    np.random.seed(6)
    (X, Y) = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    X1 = np.c_[np.ones((X.shape[0])), X]
    return X,X1,Y


def svm_training(data_dict, learning_rate, max_feature_value):
    """
    训练数据
    """
    w = []  # weights 2 dimensional vector
    b = []  # bias
    b_step_size = 2
    b_multiple = 5
    w_optimum = max_feature_value * 0.5
    length_Wvector = {}
    transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

    for lrate in learning_rate:
        w = np.array([w_optimum, w_optimum])
        optimized = False
        while not optimized:
            # b=[-maxvalue to maxvalue] we wanna maximize the b values so check for every b value
            for b in np.arange(-1 * (max_feature_value * b_step_size), max_feature_value * b_step_size, lrate * b_multiple):
                for transformation in transforms:
                    w_t = w * transformation
                    correctly_classified = True
                    for yi in data_dict:
                        for xi in data_dict[yi]:
                            # we want  yi*(np.dot(w_t,xi)+b) >= 1 for correct classification
                            if yi * (np.dot(w_t,xi) + b) < 1:
                                correctly_classified = False

                    if correctly_classified:
                        length_Wvector[np.linalg.norm(w_t)] = [w_t, b]  # store w, b for minimum magnitude

            if w[0] < 0:
                optimized = True
            else:
                w = w - lrate

        norms = sorted([n for n in length_Wvector])
        minimum_wlength = length_Wvector[norms[0]]
        w = minimum_wlength[0]
        b = minimum_wlength[1]
        w_optimum = w[0] + lrate * 2

    return w,b


def visualize(X1, w, b, max_feature_value,min_feature_value):
    """
    可视化展示
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=Y)

    datarange = (min_feature_value * 0.9, max_feature_value * 1.)
    hyp_x_min = datarange[0]
    hyp_x_max = datarange[1]

    # hyperplane = x.w+b
    # v = x.w+b
    # psv = 1
    # nsv = -1
    # dec = 0

    # (w.x+b) = 1
    # positive support vector hyperplane
    psv1 = hyperplane_value(hyp_x_min, w, b, 1)
    psv2 = hyperplane_value(hyp_x_max, w, b, 1)
    ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

    # (w.x+b) = -1
    # negative support vector hyperplane
    nsv1 = hyperplane_value(hyp_x_min, w, b, -1)
    nsv2 = hyperplane_value(hyp_x_max, w, b, -1)
    ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

    # (w.x+b) = 0
    # positive support vector hyperplane
    db1 = hyperplane_value(hyp_x_min, w, b, 0)
    db2 = hyperplane_value(hyp_x_max, w, b, 0)
    ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

    plt.axis([-5, 10, -12, -1])
    plt.show()


def hyperplane_value(x, w, b, v):
    return (-w[0] * x - b + v) / w[1]


def predict(features,w,b):
    """
    预测
    """
    # sign( x.w+b )
    dot_result = np.sign(np.dot(np.array(features), w) + b)
    return dot_result.astype(int)


if __name__ == '__main__':
    # 创建数据并绘图展示
    X, X1, Y = load_data()

    # 分为两组
    postiveX=[]
    negativeX=[]
    for i,v in enumerate(Y):
        if v==0:
            negativeX.append(X[i])
        else:
            postiveX.append(X[i])

    # 数据字典
    data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)}
    max_feature_value = float('-inf')
    min_feature_value = float('+inf')
    for yi in data_dict:
        if np.amax(data_dict[yi]) > max_feature_value:
            max_feature_value = np.amax(data_dict[yi])
        if np.amin(data_dict[yi]) < min_feature_value:
            min_feature_value = np.amin(data_dict[yi])
    learning_rate = [max_feature_value * 0.1, max_feature_value * 0.01, max_feature_value * 0.001, ]

    # 训练数据
    w,b = svm_training(data_dict, learning_rate, max_feature_value)

    # 绘制
    visualize(X1, w, b, max_feature_value,min_feature_value)

    # 预测
    data_predict = []
    for xi in X:
        data_predict.append(predict(xi,w,b))
    data_predict = np.array(data_predict).astype(int)

    # 计算误差
    data_origin = []
    for yi in Y:
        if yi==0:
            data_origin.append(-1)
        else :
            data_origin.append(1)
    data_origin = np.array(data_origin).astype(int)

    error = np.sum((data_predict-data_origin)**2)
    print("total error is ", error)
