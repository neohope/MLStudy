#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Support Vector Machines, SVM, 支持向量机
Sequential Minimal Optimization, SMO, 序列最小优化
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(fileName):
    """
    Description:
        l对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


class opt_struct:
    """
    建立的数据结构来保存所有的重要值
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        Args:
            dataMatIn    数据集
            classLabels  类别标签
            C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
                控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
                可以通过调节该参数达到不同的结果。
            toler   容错率
            kTup    如果kTup为None则没有核函数；如果不为Node，则包含核函数信息的元组, 方法可以为lin 或 rbf
        """
        # 数据集
        self.X = dataMatIn
        # 类别标签
        self.labelMat = classLabels
        # 松弛变量
        self.C = C
        # 容错率
        self.tol = toler

        # 数据的行数
        self.m = np.shape(dataMatIn)[0]
        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 拉格朗日乘子
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 模型的常量值
        self.b = 0

        # 如果kTup为None则没有核函数；
        # 如果不为Node，则包含核函数信息的元组, 方法可以为lin 或 rbf
        if(kTup!=None):
            self.K = np.mat(np.zeros((self.m, self.m)))
            for i in range(self.m):
                self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)


def kernelTrans(X, A, kTup):
    """
    核转换函数，有核函数才会用到
    Args:
        X     dataMatIn数据集
        A     dataMatIn数据集的第i行的数据
        kTup  核函数的信息
    Returns:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:
        # m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Kernel can just be lin or rbf')
    return K


def smop_math(dataMatIn, classLabels, C, toler, maxIter, kTup=None):
    """
    完整SMO算法外循环
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
        kTup    包含核函数信息的元组
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = opt_struct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环小于maxIter次
    # 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS, kTup!=None)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 对已存在 alpha对，选出非边界的alpha值，进行优化。
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS, kTup!=None)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找
        # 如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def smop(dataMatIn, classLabels, C, toler, maxIter, kTup=None):
    """
    SMO外循环
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
        kTup    包含核函数信息的元组
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = opt_struct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环小于maxIter次
    # 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        if entireSet:
            alphaPairsChanged += np.sum(innerL(i, oS, kTup!=None) for i in range(oS.m))
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            alphaPairsChanged += np.sum(innerL(i, oS, kTup!=None) for i in nonBoundIs)
        iter += 1

        # 如果找到alpha对，就优化非边界alpha值
        # 否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def innerL(i, oS, hasKernal):
    """innerL
    SMO内循环
    Args:
        i   具体的某一行
        oS  optStruct对象
    Returns:
        0   找不到最优的值
        1   找到了最优的值，并且oS.Cache到缓存中
    """

    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i, hasKernal)

    # KKT约束条件
    # 检验训练样本(xi, yi)是否满足KKT条件
    # yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    # yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # yi*f(i) <= 1 and alpha = C (between the boundary)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei, hasKernal)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        if(hasKernal):
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        else:
            eta = oS.X[i] - oS.X[j]
            eta = - eta * eta.T
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j, hasKernal)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i, hasKernal)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        if(hasKernal):
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        else:
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[i].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i] * oS.X[j].T
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[j].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j] * oS.X[j].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def calcEk(oS, k, hasKernal):
    """
    求 Ek误差：预测值-真实值的差
    Args:
        oS  optStruct对象
        k   具体的某一行

    Returns:
        Ek  预测结果与真实结果比对，计算误差Ek
    """
    if(hasKernal):
        fXk = np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b
    else :
        fXk = np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k].T) + oS.b

    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei, hasKernal):
    """
    返回最优的j和Ej
    选择第内循环alpha的alpha值
    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
    Args:
        i   具体的第i一行
        oS  optStruct对象
        Ei  预测结果与真实结果比对，计算误差Ei
    Returns:
        j  随机选出的第j一行
        Ej 预测结果与真实结果比对，计算误差Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]

    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue
            # 求 Ek误差：预测值-真实值的差
            Ek = calcEk(oS, k, hasKernal)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)
        # 求 Ek误差：预测值-真实值的差
        Ej = calcEk(oS, j, hasKernal)
    return j, Ej


def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = np.random.randint(0, m - 1)
    return j


def clipAlpha(aj, H, L):
    """
    调整aj的值，使L<=aj<=H
    Args:
        aj  目标值
        H   最大值
        L   最小值
    Returns:
        aj  目标值
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj


def updateEk(oS, k, hasKernal):
    """
    计算误差值并存入缓存中。
    Args:
        oS  optStruct对象
        k   某一列的行号
    """
    # 误差：预测值-真实值的差
    Ek = calcEk(oS, k, hasKernal)
    oS.eCache[k] = [1, Ek]


def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集
    Returns:
        w  回归系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i].T)
    return w


def hyperplane_value(x, w, b, v):
    return (-w[0] * x - b + v) / w[1]


def plotfig_SVM(xArr, yArr, ws, b, alphas, hasKernal=False):
    """
    绘制SVM
    """

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = np.array(b)[0]
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    if(not hasKernal):
        x = np.arange(-1.0, 10.0, 0.1)

        hyp_x_min = x[0]
        hyp_x_max = x[-1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane_value(hyp_x_min, ws, b, 1)
        psv2 = hyperplane_value(hyp_x_max, ws, b, 1)
        ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane_value(hyp_x_min, ws, b, -1)
        nsv2 = hyperplane_value(hyp_x_max, ws, b, -1)
        ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane_value(hyp_x_min, ws, b, 0)
        db2 = hyperplane_value(hyp_x_max, ws, b, 0)
        ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

    for i in range(np.shape(yMat[0])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()
