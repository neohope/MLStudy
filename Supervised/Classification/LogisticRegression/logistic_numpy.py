#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归
numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def createData():
    """
    构建测试数据
    创建100个点，分为两组
    在X上面，增加一列1
    """
    np.random.seed(6)
    (X,Y) = make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.05,random_state=20)
    X1 = np.c_[np.ones((X.shape[0])),X]
    return X1,Y


def sigmoid(x):
    """
    数据归一化处理，将数据
    """
    return float(1.0 / float((1.0 + np.exp(-1.0*x))))


def test_sigmoid():
    """
    展示sigmoid函数
    """
    sx=range(-10,10)
    sy=[]
    for i in sx:
        sy.append(sigmoid(i))
    plt.plot(sx,sy)
    plt.show()


def predict(X1,W):
    """
    根据系数，预测结果
    """
    predicted_y = []
    for x in X1:
        # ϴ0+ϴ1*X
        logit = x.dot(W)
        predicted_y.append(sigmoid(logit))
    return np.array(predicted_y)


def predictto01(yhat):
    """
    将数据划分为0，1
    """
    for i,v in enumerate(yhat):
        if v >=0.56:
            yhat[i]=1
        else:
            yhat[i]=0

    return yhat.astype(int)


def cost_function(predicted_y, Y):
    """
    预测结果评估函数
    """
    error = (-Y * np.log(predicted_y)) - ((1 - Y) * np.log(1 - predicted_y))
    cf = (1 / X1.shape[0]) * sum(error)
    return cf, error


def gradient_descent(X1, Y, lrate, epochs, W):
    """
    梯度下降
    """
    total_expected_error = float("inf")
    errorlist = []
    finalepoch = 0

    for epoch in range(epochs):
        predictedY = predict(X1, W)
        total_error, error = cost_function(predictedY, Y)

        # d/dϴ =error*x.T
        # 梯度下降
        gradient = X1.T.dot(error) / X1.shape[0]

        # 每十次记录一次错误情况
        if epoch % 10 == 0:
            errorlist.append(total_error)
            finalepoch += 1

        # 结束训练
        if (total_expected_error < total_error):
            return errorlist, finalepoch
        total_expected_error = total_error

        # 进行系数修正
        for (i, w) in enumerate(gradient):
            W[i] += float(-lrate) * w

    return errorlist, finalepoch


if __name__ == '__main__':
    # 测试sigmoid函数
    test_sigmoid()

    # 创建数据并绘图展示
    X1, Y= createData()
    plt.scatter(X1[:,1],X1[:,2],marker='o',c=Y)
    plt.show()
    plt.scatter(X1[:,1],Y,marker='o',c=Y)
    plt.show()

    # 初始化系数
    W=np.random.uniform(size=X1.shape[1])

    # 梯度下降
    total_error,finalepoch=gradient_descent(X1,Y,0.001,100,W)

    #展示错误下降情况并绘图展示
    plt.plot(range(finalepoch),total_error)
    plt.xlabel("epochs in 10's")
    plt.ylabel("error")
    plt.show()

    # 预测结果并绘图展示
    yhat= predictto01(predict(X1, W))
    plt.scatter(X1[:,1],X1[:,2],marker='o',c=yhat)
    plt.show()

    #输出error和acuracy
    error=sum((yhat-Y)**2)
    print(error)
    accuracy=1-(error/100)
    print(accuracy)
