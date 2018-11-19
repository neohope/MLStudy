#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
验证numpy安装没有问题
"""

import numpy as np
from numpy import linalg as la

"""
NumPy存在2中不同的数据类型:
    1. 矩阵 matrix
    2. 数组 array
"""

if __name__ == '__main__':
    # 生成一个 4*4 的随机Array
    randArray = np.random.rand(4, 4)
    print('randArray=(%s) \n' % type(randArray), randArray)

    # Array转换为Matrix
    randMat = np.mat(randArray)
    print('randMat=(%s) \n' % type(randMat), randMat)

    # Matrix转换为Array
    ArrRandMat = randMat.A
    print('ArrRandMat=(%s) \n' % type(ArrRandMat), ArrRandMat)

    # 矩阵求逆
    invRandMat = randMat.I
    print('invRandMat=(%s) \n' % type(invRandMat), invRandMat)
    # 矩阵和逆矩阵求积
    myEye = randMat*invRandMat
    # 计算误差
    print('myEye=(%s) \n' % type(myEye), myEye-np.eye(4))

    # 矩阵转置
    TraRandMat = randMat.T
    print('TraRandMat=(%s) \n' % type(TraRandMat), TraRandMat)

    # 特征值和特征向量
    myEig = la.eig(randMat)
    print('myEig[0]=(%s) \n' % type(myEig[0]), myEig[0])
    print('myEig[1]=(%s) \n' % type(myEig[1]), myEig[1])

    # 矩阵行列式
    myDet = la.det(randMat)
    print('myDet=(%s) \n' % type(myDet), myDet)
