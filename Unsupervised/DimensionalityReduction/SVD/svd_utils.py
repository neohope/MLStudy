#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Singular Value Decomposition，SVD
工具类
"""

from numpy import linalg as la
import numpy as np


def ecludSim(inA, inB):
    """
    欧氏距离相似度
    """
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    """
    皮尔逊相似度
    """
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0

    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    余弦相似度
    如果夹角为90度，相似度为0；
    如果两个向量的方向相同，相似度为1.0
    """
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)


def standEst(dataMat, user, simMeas, item):
    """
    基于物品相似度的推荐引擎
    计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分
    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 得到数据集中的物品数目
    n = np.shape(dataMat)[1]

    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0

    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]

        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue

        # 寻找两个用户都评级的物品
        # 变量 overLap 给出的是两个物品当中已经被评分的那个元素的索引ID
        # logical_and 计算x1和x2元素的真值。
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]

        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度
        # userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0

    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal


def svdEst(dataMat, user, simMeas, item):
    """
    基于物品相似度的推荐引擎
    基于SVD的评分估计
    先做降维运算，然后推荐
    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """

    # 物品数目
    n = np.shape(dataMat)[1]

    # 对数据集进行SVD分解
    simTotal = 0.0
    ratSimTotal = 0.0

    # 奇异值分解
    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U, Sigma, VT = la.svd(dataMat)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[: 4])

    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I

    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue

        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)

        # 对相似度不断累加求和
        simTotal += similarity

        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal/simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    推荐引擎，产生了最高的N个推荐结果。
    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        estMethod       使用的推荐算法
    Returns:
        返回最终 N 个推荐结果
    """

    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]

    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'

    # 物品的编号和评分值
    itemScores = []

    # 在未评分物品上进行循环
    for item in unratedItems:
        # 获取 item 该物品的评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]


def analyse_data(Sigma, loopNum=20):
    """
    分析Sigma
    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])
        print('属性：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))
