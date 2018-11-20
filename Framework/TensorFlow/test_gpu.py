"""
gpu test
"""

import tensorflow as tf
import numpy as np


def test_gpu0():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess.run(c))


def test_gpu1():
    with tf.device('/gpu:1'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        print(sess.run(c))


def test_multi_gpu():
    c = []

    # GPU做运算
    for d in ['/device:GPU:0', '/device:GPU:1']:
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))

    # CPU求和
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(sum))


if __name__ == '__main__':
    test_gpu0()
    test_gpu1()
    test_multi_gpu()