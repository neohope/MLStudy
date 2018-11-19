#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
graph test
"""

import tensorflow as tf


def test():
    """
    default
    """
    value1 = tf.constant([1., 2.])
    value2 = tf.Variable([3., 4.])
    result = value1 * value2
    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        print(sess.run(result))


def test_graph():
    """
    指定graph
    """
    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.constant([1., 2.])
        value2 = tf.Variable([3., 4.])
        result = value1 * value2

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run(session=sess)
        print(result.eval())


def test_gpu():
    """
    指定gpu
    """
    value1 = tf.constant([1., 2.])
    value2 = tf.Variable([3., 4.])
    result = value1 * value2
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            tf.global_variables_initializer().run(session=sess)
            print(sess.run(result))


def test_graph_gpu():
    """
    指定gpu和graph
    """
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:0"):
            value1 = tf.constant([1., 2.])
            value2 = tf.Variable([3., 4.])
            result = value1 * value2

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run(session=sess)
        print(result.eval())


if __name__ == '__main__':
    test()
    test_graph()
    # test_gpu()
    # test_graph_gpu()
