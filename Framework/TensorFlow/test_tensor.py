#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
tensor test
"""

import tensorflow as tf
import numpy as np


def test_const():
    a = tf.constant(2)
    b = tf.constant(3)
    with tf.Session() as sess:
        print(sess.run(a + b))


def test_variable():
    v1 = tf.Variable(10)
    v2 = tf.Variable(5)

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        print(sess.run(v1 + v2))


def test_placeholder():
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)
    add = tf.add(a, b)
    mul = tf.multiply(a, b)
    with tf.Session() as sess:
        print(sess.run(add, feed_dict={a: 2, b: 3}))
        print(sess.run(mul, feed_dict={a: 2, b: 3}))


def test_matrix():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    with tf.Session() as sess:
        print(product.eval())


def test_data_type():
    a = np.array([2, 3], dtype=np.int32)
    b = np.array([4, 5], dtype=np.int32)
    c = tf.add(a, b)
    with tf.Session() as sess:
        print(sess.run(c))


if __name__ == '__main__':
    test_const()
    test_variable()
    test_placeholder()
    test_matrix()
    test_data_type()
