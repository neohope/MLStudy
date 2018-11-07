#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归工具类
"""

import numpy as np


def load_data(fileName):
    """
    Desc：
        该函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数
    Args:
        fileName 文件名
    Returns:
        dataMat 每一行的数据集array类型
    Raises:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)
    return dataMat


def regTreeEval(model, inDat):
    """
    Desc:
        对 回归树 进行预测
    Args:
        model -- 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
        inDat -- 输入的测试数据
    Returns:
        float(model) -- 将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


def regLeaf(dataSet):
    # 返回每一个叶子结点的均值
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    # 计算总方差=方差*样本数
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def modelTreeEval(model, inDat):
    """
    Desc:
        对 模型树 进行预测
    Args:
        model -- 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型
        inDat -- 输入的测试数据
    Returns:
        float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(X * model)


def modelLeaf(dataSet):
    """
    Desc:
        当数据不再需要切分的时候，生成叶节点的模型。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回得到的 回归系数ws
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    Desc:
        在给定数据集上计算误差。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Description：
        递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程
    Args:
        dataSet      加载的原始数据集
        leafType     建立叶子点的函数
        errType      误差计算函数
        ops=(1, 4)   [容许误差下降值，切分的最少样本数]
    Returns:
        retTree    决策树最后的结果
    """
    # 选择最好的切分方式： feature索引值，最优切分值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)

    # 如果 splitting 达到一个停止条件，那么返回 val
    if feat is None:
        return val

    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)

    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Description：
        用最佳方式切分数据集 和 生成相应的叶节点
    Args:
        dataSet   加载的原始数据集
        leafType  建立叶子点的函数
        errType   误差计算函数(求总方差)
        ops       [容许误差下降值，切分的最少样本数]。
    Returns:
        bestIndex feature的index坐标
        bestValue 切分的最优值
    Raises:
    """

    # 因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
    tolS = ops[0]
    tolN = ops[1]

    #如果数据集的最后一列所有值相等就退出
    # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分。
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        #  exit cond 1
        return None, leafType(dataSet)

    # 计算行列值
    m, n = np.shape(dataSet)
    # 无分类误差的总方差和
    S = errType(dataSet)
    # inf 正无穷大
    bestS, bestIndex, bestValue = np.inf, 0, 0

    # 循环处理每一列对应的feature值
    for featIndex in range(n-1):
        # 下面的一行表示的是将某一列全部的数据转换为行，然后设置为list形式
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 判断二元切分的方式的元素误差是否符合预期
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

    # 如果集合的 size 小于 tolN，集合过小，也不划分，产生叶节点
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestIndex, bestValue


def binSplitDataSet(dataSet, feature, value):
    """
    Description：
        在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
    Args:
        dataMat 数据集
        feature 待切分的特征列
        value 特征列要比较的值
    Returns:
        mat0 小于等于 value 的数据集在左边
        mat1 大于 value 的数据集在右边
    Raises:
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    Desc:
        调用 treeForeCast ，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值矩阵
    """
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    Desc:
        对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值
    """
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def isTree(obj):
    """
    Desc:
        测试输入变量是否是一棵树,即是否是一个字典
    Args:
        obj -- 输入变量
    Returns:
        返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
    """
    return (type(obj).__name__ == 'dict')


def linearSolve(dataSet):
    """
    Desc:
        将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
    Args:
        dataSet -- 输入数据
    Returns:
        ws -- 执行线性回归的回归系数
        X -- 格式化自变量X
        Y -- 格式化目标变量Y
    """
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    #Y = np.mat(np.ones((m, 1)))
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
