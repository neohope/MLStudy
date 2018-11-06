#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
工具类
"""

import numpy as np


def load_data(fileName):
    """
    Description：
        加载数据
    Returns：
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的分类标签，即类别标签
    """
    # 获取样本特征的总数
    numFeat = len(open(fileName).readline().split('\t')) - 1

    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # 读取每一行
        lineArr = []
        # 删除一行中以tab分隔的数据前后的空白符号
        curLine = line.strip().split('\t')
        # 加载每个特征值
        for i in range(numFeat):
            # 将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
            lineArr.append(float(curLine[i]))
        # 将测试数据的输入数据部分存储到dataMat 的List中
        dataMat.append(lineArr)
        # 将每一行的最后一个数据，即类别，或者叫目标变量存储到labelMat List中
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    Description：
        测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    Args：
        testArr：测试所用的所有样本点
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量
        k：控制核函数的衰减速率
    Returns：
        yHat：预测点的估计值
    '''
    # 得到样本点的总数
    m = np.shape(testArr)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)

    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    Description：
        局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
    Args：
        testPoint：样本点
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量
        k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
    Returns:
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点
    Notes:
        这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
        理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
        关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
        也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    '''
    # mat() 函数是将array转换为矩阵的函数， mat().T 是转换为矩阵之后，再进行转置操作
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # 获得xMat矩阵的行数
    m = np.shape(xMat)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = np.mat(np.eye((m)))

    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j, :]
        # k控制衰减的速度
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def ridgeTest(xArr, yArr):
    '''
    Desc：
        函数 ridgeTest() 用于在一组λ上测试结果
    Args：
        xArr：样本数据的特征，即 feature
        yArr：样本数据的类别标签，即真实数据
    Returns：
        wMat：将所有的回归系数输出到一个矩阵并返回
    '''

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # 计算Y的均值
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean

    # X均值
    xMeans = np.mean(xMat, 0)
    # X方差
    xVar = np.var(xMat, 0)
    # 所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMeans) / xVar

    # 创建30 * m 的全部数据为0 的矩阵
    num_test_points = 30
    wMat = np.zeros((num_test_points, np.shape(xMat)[1]))
    for i in range(num_test_points):
        # exp() 返回 e^x
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    Desc：
        这个函数实现了给定 lambda 下的岭回归求解。
        如果数据的特征比样本点还多，就不能再使用上面介绍的的线性回归和局部现行回归了，因为计算 (xTx)^(-1)会出现错误。
        如果特征比样本点还多（n > m），也就是说，输入数据的矩阵x不是满秩矩阵。非满秩矩阵在求逆时会出现问题。
        为了解决这个问题，我们下边讲一下：岭回归，这是我们要讲的第一种缩减方法。
    Args：
        xMat：样本的特征数据，即 feature
        yMat：每个样本对应的类别标签，即目标变量，实际值
        lam：引入的一个λ值，使得矩阵非奇异
    Returns：
        经过岭回归公式计算得到的回归系数
    '''

    xTx = xMat.T * xMat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def rssError(yArr, yHatArr):
    '''
    Desc:
        计算分析预测误差的大小
    Args:
        yArr：真实的目标变量
        yHatArr：预测得到的估计值
    Returns:
        计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr) ** 2).sum()
