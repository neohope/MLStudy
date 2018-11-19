#!/usr/bin/python
# -*- coding:utf-8 -*-

"""

"""

import tensorflow as tf


def test_matrix():
    x = tf.constant([[2, 5, 3, -5],
                     [0, 3, -2, 5],
                     [4, 3, 5, 3],
                     [6, 1, 4, 0]])

    y = tf.constant([[4, -7, 4, -3, 4],
                     [6, 4, -7, 4, 7],
                     [2, 3, 2, 1, 4],
                     [1, 5, 5, 5, 2]])

    with tf.Session() as sess:
        # 矩阵转置
        print(tf.transpose(x).eval())

        # 逆矩阵
        print(tf.matrix_inverse(tf.to_float(x)).eval())

        # 特征值，特征向量
        eigenvaules, eigenvectors = tf.self_adjoint_eig(tf.to_float(x))
        print("eigenvaules=", eigenvaules)
        print("eigenvectors=", eigenvectors)

        # 矩阵的行列式
        print(tf.matrix_determinant(tf.to_float(x)).eval())

        # 矩阵乘法
        print(tf.matmul(x, y).eval())

        # 矩阵除法
        r = tf.matrix_solve(tf.to_float(x), [[1], [1], [1], [1]]).eval()
        print(r)
        # print(tf.matmul(tf.to_float(x), r).eval())


def test_reduction():
    x = tf.constant([[1, 2, 3],
                     [3, 2, 1],
                     [-1, -2, -3]])
    with tf.Session() as sess:
        # 乘法
        print(tf.reduce_prod(x).eval())
        # 按行乘法
        print(tf.reduce_prod(x, reduction_indices=1).eval())
        # 按行最小值
        print(tf.reduce_min(x, reduction_indices=1).eval())
        # 按行最大值
        print(tf.reduce_max(x, reduction_indices=1).eval())
        # 按行平均值
        print(tf.reduce_mean(x, reduction_indices=1).eval())

    boolean_tensor = tf.constant([[True, False, True],
                                  [False, False, True],
                                  [True, False, False]])
    with tf.Session() as sess:
        # 按行And
        print(tf.reduce_all(boolean_tensor, reduction_indices=1).eval())
        # 按行Or
        print(tf.reduce_any(boolean_tensor, reduction_indices=1).eval())


def test_segment():
    seg_ids = tf.constant([0, 1, 1, 2, 2])
    x = tf.constant([[2, 5, 3, -5],
                     [0, 3, -2, 5],
                     [4, 3, 5, 3],
                     [6, 1, 4, 0],
                     [6, 1, 4, 0]])
    with tf.Session() as sess:
        # 按seg_ids进行加法
        print(tf.segment_sum(x, seg_ids).eval())
        # 按seg_ids进行乘法
        print(tf.segment_prod(x, seg_ids).eval())
        # 按seg_ids进行min运算
        print(tf.segment_min(x, seg_ids).eval())
        # 按seg_ids进行max运算
        print(tf.segment_max(x, seg_ids).eval())
        # 按seg_ids进行mean运算
        print(tf.segment_mean(x, seg_ids).eval())


def test_sequence():
    x = tf.constant([[2, 5, 3, -5],
                     [0, 3, -2, 5],
                     [4, 3, 5, 3],
                     [6, 1, 4, 0]])
    with tf.Session() as sess:
        # 返回各列最小值的索引
        print(tf.argmin(x, 0).eval())
        # 返回各行最大值的索引
        print(tf.argmax(x, 1).eval())

    boolx = tf.constant([[True, False], [False, True]])
    with tf.Session() as sess:
        # 返回 Tensor 为 True 的位置
        print(tf.where(boolx).eval())

    listx = tf.constant([1, 2, 5, 3, 4, 5, 6, 7, 8, 3, 2])
    with tf.Session() as sess:
        # 去除重复数据
        print(tf.unique(listx)[0].eval())


if __name__ == '__main__':
    #test_matrix()
    #test_reduction()
    #test_segment()
    test_sequence()
