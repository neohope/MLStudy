#!/usr/bin/python
# -*- coding:utf-8 -*-

"""

"""

import numpy as np
from numpy import linalg as la


def load_data():
    """
    加载数据
    """
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


if __name__ == "__main__":
    # 对矩阵进行SVD分解
    Data = load_data()
    U, Sigma, VT = la.svd(Data, full_matrices=False)

    # 输出果
    print('U:\n', U.shape)
    print('Sigma:\n', Sigma.shape)
    print('VT:\n', VT.shape)

    # 还原矩阵
    SigmaX = np.mat(np.eye(len(Sigma)) * Sigma)
    print("Data:\n", (U*SigmaX*VT).astype(int))
