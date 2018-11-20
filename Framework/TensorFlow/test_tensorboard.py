#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
tensorboard --logdir="PATH_TO_OUTPUT/output/hello_graph"
tensorboard --logdir="PATH_TO_OUTPUT/output/name_scope"
打开浏览器 http://localhost:6006
"""

import tensorflow as tf


def test1():
    """
    a=5
    b=3
    c=a*b=15
    d=a+b=8
    e=c+d=23
    """
    a = tf.constant(5, name="input_a")
    b = tf.constant(3, name="input_b")
    c = tf.multiply(a, b, name="mul_c")
    d = tf.add(a, b, name="add_d")
    e = tf.add(c, d, name="add_e")

    with tf.Session() as sess:
        print(sess.run(e))
        writer = tf.summary.FileWriter("output/hello_graph", sess.graph)


def test2():
    """
    a=[5,3]
    b=a*b=15
    c=a+b=8
    d=c+d=23
    """
    a = tf.constant([5, 3], name="input_a")
    b = tf.reduce_prod(a, name="prod_b")
    c = tf.reduce_sum(a, name="sum_c")
    d = tf.add(c, b, name="add_d")

    with tf.Session() as sess:
        print(sess.run(d))
        writer = tf.summary.FileWriter("output/hello_graph", sess.graph)


def test3():
    # scope_a
    # a=1+2=3
    # b=a*3=9
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="A_add")
        b = tf.multiply(a, 3, name="A_mul")

    # scope_b
    # c=4+5=9
    # d=c*6=54
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="B_add")
        d = tf.multiply(c, 6, name="B_mul")

    # e=b+d=63
    e = tf.add(b, d, name="output")

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('output/name_scope', graph=tf.get_default_graph())


if __name__ == '__main__':
    # test1()
    # test2()
    test3()
