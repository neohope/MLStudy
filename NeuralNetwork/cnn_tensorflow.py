#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CNN/Convnets/Convolutional neural networks
tensorflow
"""

from keras.utils import np_utils
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np


def train(X_train,Y_train,X_test,Y_test):
    """
    训练
    """
    X = tf.placeholder("float", [None, 32, 32, 3])  # as our dataset
    Y = tf.placeholder("float", [None, 10])

    p_keep_conv = tf.placeholder("float")  # for dropouts as percentage
    p_keep_hidden = tf.placeholder("float")

    W_C1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    W_C2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
    W_C3 = init_weights([3, 3, 64, 128])  # 3x3x64 conv, 128 outputs
    W_FC = init_weights([128 * 4 * 4, 625])  # FC 128*4*4 inputs, 625 outputs
    W_O = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    # 设置MSProp优化器
    Y_pred = model(X, W_C1, W_C2, W_C3, W_FC, W_O, p_keep_conv, p_keep_hidden)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)  # compute mean cross entropy (softmax is applied internally)
    predict_op = tf.argmax(Y_pred, 1)  # at predict time, evaluate the argmax of the logistic regression

    # reshape for tf
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)
    epochs = 50

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train) + 1, 128)):
                session.run(optimizer,feed_dict={X: X_train[start:end], Y: Y_train[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})

            #每10次输出一次情况
            if epoch % 10 == 0:
                accuracy = np.mean(np.argmax(Y_test, axis=1) == session.run(predict_op,feed_dict={X: X_test, p_keep_conv: 1.0,p_keep_hidden: 1.0}))
                print("epoch : {} and accuracy : {}".format(epoch, accuracy))
                print("testing labels for test data")
                print(session.run(predict_op, feed_dict={X: X_test, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        print("Final accuracy : {}".format(np.mean(np.argmax(Y_test, axis=1) == session.run(predict_op,feed_dict={X: X_test,p_keep_conv: 1.0,p_keep_hidden: 1.0}))))


def init_weights(shape):
    """
    初始化权重系数
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W_C1, W_C2, W_C3, W_FC, W_O, p_keep_conv, p_keep_hidden):
    """
    调整权重系数
    """
    C1 = tf.nn.relu(tf.nn.conv2d(X, W_C1, strides=[1, 1, 1, 1], padding="SAME"))
    P1 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 1st pooling layer shape =(?,14,14,32)
    D1 = tf.nn.dropout(P1, p_keep_conv)  # 1st dropout at conv
    C2 = tf.nn.relu(tf.nn.conv2d(D1, W_C2, strides=[1, 1, 1, 1], padding="SAME"))  # 2nd convoultion layer shape=(?, 14, 14, 62)
    P2 = tf.nn.max_pool(C2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 2nd pooling layer shape =(?,7,7,64)
    D2 = tf.nn.dropout(P2, p_keep_conv)  # 2nd dropout at conv
    C3 = tf.nn.relu(tf.nn.conv2d(D2, W_C3, strides=[1, 1, 1, 1], padding="SAME"))  # 3rd convoultion layer shape=(?, 7, 7, 128)
    P3 = tf.nn.max_pool(C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 3rd pooling layer shape =(?,4,4,128)
    P3 = tf.reshape(P3, [-1, W_FC.get_shape().as_list()[0]])  # reshape to (?, 2048)
    D3 = tf.nn.dropout(P3, p_keep_conv)  # 3rd dropout at conv
    FC = tf.nn.relu(tf.matmul(D3, W_FC))
    FC = tf.nn.dropout(FC, p_keep_hidden)  # droput at fc
    output = tf.matmul(FC, W_O)
    return output


if __name__ == '__main__':
    # 加载测试数据，一堆32*32*3的图片
    # 10类
    # 50000训练数据
    # 10000测试数据
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    print(X_train.shape)
    print(X_test.shape)
    print(X_train[0][0])

    # 将数据转换为0-1的浮点数
    X_train=X_train/255.0
    X_test=X_test/255.0
    print(X_train[0][0])

    # 将Y转换为标签矩阵
    # 属于哪一类，哪一列为1，其余为0
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    print(Y_train[0])

    # 训练
    # 跑了5个迭代，准确率0.6196，而且每次迭代上升还比较明显
    train(X_train,Y_train,X_test,Y_test)
