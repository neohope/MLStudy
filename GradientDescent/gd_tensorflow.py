#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
梯度下降算法，进行回归分析
tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def createData(points):
    """
    创建100个符合sin分布的点
    并引入一些噪声
    """
    X=np.linspace(-3,3,100)
    np.random.seed(6)
    Y=np.sin(X)+np.random.uniform(-0.5,0.5,points)
    return X,Y


if __name__ == '__main__':
    # 创建数据
    X, Y = createData(100)

    # 设置为一元线性回归
    # YF = WF*XF + BF
    XF = tf.placeholder(tf.float32)
    YF = tf.placeholder(tf.float32)
    tf.set_random_seed(5)
    WF = tf.Variable(tf.random_normal([1]), name="weights")
    BF = tf.Variable(tf.random_normal([1]), name='bias')
    YF_Pred = tf.add(tf.multiply(XF, WF), BF)

    # 设置验证方式
    error = tf.square(YF_Pred - YF)
    f_error = tf.reduce_sum(error)/(100-1)

    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(f_error)

    # 初始化绘图
    plt.plot(X, Y, 'ro')
    plt.axis([-4, 4, -2.0, 2.0])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epochs = 500
        loss_expected = 0

        #开始训练
        for epoch in range(epochs):

            #将训练数据一次喂给优化器
            for (x_point, y_point) in zip(X, Y):
                session.run(optimizer, {XF: x_point, YF: y_point})

            #用同样的数据，检验优化结果
            loss_per_epoch = session.run(f_error, {XF: X, YF: Y})

            # 每10次训练，增加一条线，每次修正后线的颜色都会加深
            if epoch%10 == 0:
                plt.axis([-4, 4, -2.0, 2.0])
                plt.plot(X, YF_Pred.eval(feed_dict={XF: X}, session=session),'b', alpha=epoch/epochs)

            # 相邻两次训练没有明显提升，则结束训练
            if np.abs(loss_expected - loss_per_epoch) < 0.000001:
                break
            loss_expected = loss_per_epoch

    # 展示拟合结果
    plt.show()

