#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Dimensionality Reduction
Principal Component Analysis，PCA，主成分分析
"""

from Unsupervised.DimensionalityReduction.PCA import pca_utils


if __name__ == "__main__":
    # 加载数据，并转化数据类型为float
    dataMat = pca_utils.load_data('../../../Data/PrincipalComponentAnalysis/testSet.txt')

    # 分析数据
    pca_utils.analyse_data(dataMat)

    # 只需要1个特征向量
    lowDmat, reconMat = pca_utils.pca(dataMat, 1)
    pca_utils.show_picture(dataMat, reconMat)

    # 只需要2个特征向量，和原始数据一致，没任何变化
    lowDmat, reconMat = pca_utils.pca(dataMat, 2)
    pca_utils.show_picture(dataMat, reconMat)
