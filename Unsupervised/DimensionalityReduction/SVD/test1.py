#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
测试距离计算函数
"""

import numpy as np
from Unsupervised.DimensionalityReduction.SVD import svd_utils


def load_data():
    """
    加载数据
    """
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]


if __name__ == "__main__":
    # 加载数据
    myMat = np.mat(load_data())

    # 计算欧氏距离
    print(svd_utils.ecludSim(myMat[:, 0], myMat[:, 4]))
    print(svd_utils.ecludSim(myMat[:, 0], myMat[:, 0]))

    # 计算余弦相似度
    print(svd_utils.cosSim(myMat[:, 0], myMat[:, 4]))
    print(svd_utils.cosSim(myMat[:, 0], myMat[:, 0]))

    # 计算皮尔逊相关系数
    print(svd_utils.pearsSim(myMat[:, 0], myMat[:, 4]))
    print(svd_utils.pearsSim(myMat[:, 0], myMat[:, 0]))