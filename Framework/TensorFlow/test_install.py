#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
安装测试
"""

import tensorflow as tf

if __name__ == '__main__':
    print('Tensorflow version is', tf.__version__)
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
