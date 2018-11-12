#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Dimensionality Reduction
Principal Component Analysis，PCA，主成分分析
利用PCA对半导体制造数据降维
"""

import numpy as np
from Unsupervised.DimensionalityReduction.PCA import pca_utils


def replaceNanWithMean():
    """
    加载数据，并将数据中为NAN的数据，替换为平均值
    """
    datMat = pca_utils.load_data('../../../Data/PrincipalComponentAnalysis/secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0],i] = meanVal
    return datMat


if __name__ == "__main__":
    # 加载数据
    dataMat = replaceNanWithMean()

    # 分析数据
    pca_utils.analyse_data(dataMat)
    lowDmat, reconMat = pca_utils.pca(dataMat, 20)
    pca_utils.show_picture(dataMat, reconMat)
