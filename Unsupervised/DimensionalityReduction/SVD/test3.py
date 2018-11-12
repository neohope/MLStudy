#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
模拟餐厅推荐系统
"""

import numpy as np
from Unsupervised.DimensionalityReduction.SVD import svd_utils


def load_data():
    """
    加载数据
    """
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


if __name__ == "__main__":
    # 计算相似度的方法
    myMat = np.mat(load_data())

    # 不降维，欧氏距离相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.standEst, simMeas=svd_utils.ecludSim))

    # 降维，欧氏距离相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.svdEst, simMeas=svd_utils.ecludSim))

    # 不降维，皮尔逊相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.standEst, simMeas=svd_utils.pearsSim))

    # 降维，皮尔逊相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.svdEst, simMeas=svd_utils.pearsSim))

    # 不降维，余弦相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.standEst, simMeas=svd_utils.cosSim))

    # 降维，余弦相似度
    print(svd_utils.recommend(myMat, 1, estMethod=svd_utils.svdEst, simMeas=svd_utils.cosSim))
