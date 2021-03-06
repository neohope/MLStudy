#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
降维原理说明
"""

import numpy as np
from numpy import linalg as la


def load_data():
    """
    加载数据
    """
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


if __name__ == "__main__":
    # SVD矩阵分解
    U, Sigma, VT = la.svd(np.mat(load_data()))
    print(Sigma)                 # 计算矩阵的SVD来了解其需要多少维的特征

    Sig2 = Sigma**2              # 计算需要多少个奇异值能达到总能量的90%
    print(sum(Sig2) * 0.9)       # 计算总能量的90%

    print(sum(Sig2[: 3]))        # 前三个元素所包含的能量高于总能量的90%，后面的维度就可以不考虑了，达到了降维的目的
