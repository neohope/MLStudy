#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
基于SVD的图像压缩
"""

import numpy as np
from numpy import linalg as la
from Unsupervised.DimensionalityReduction.SVD import svd_utils


def imgLoadData(filename):
    """
    加载数据
    """
    myl = []
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    return myMat


def imgCompress(myMat, numSV, thresh=0.8):
    """
    实现图像压缩，允许基于任意给定的奇异值数目来重构图像
    Args:
        numSV       Sigma长度
        thresh      判断的阈值
    """
    # 对原始图像进行SVD分解并重构图像

    # SVD矩阵分解
    U, Sigma, VT = la.svd(myMat)

    # 分析插入的 Sigma 长度
    svd_utils.analyse_data(Sigma, 20)

    SigRecon = np.mat(np.eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]

    print("reconstructed matrix using %d singular values:" % numSV)
    printMat(reconMat, thresh)


def printMat(inMat, thresh=0.8):
    """
    打印矩阵
    """
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,end='')
            else:
                print(0,end='')
        print('')


if __name__ == "__main__":
    myMat = imgLoadData('../../../Data/SingularValueDecomposition/0_5.txt')
    print("original matrix:")
    printMat(myMat, thresh=0.8)

    # 这个效果会比较差
    imgCompress(myMat, numSV=1, thresh=0.8)

    # 后面两个差不多
    imgCompress(myMat, numSV=2, thresh=0.8)
    imgCompress(myMat, numSV=3, thresh=0.8)
