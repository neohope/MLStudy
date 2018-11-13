#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ANN神经网络Demo
"""

from NeuralNetwork.ANN import ann_fcl


if __name__ == '__main__':
    labels, data_set = ann_fcl.transpose(ann_fcl.train_data_set())
    labels=list(labels)
    data_set=list(data_set)

    net = ann_fcl.Network([8, 3, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    for i in range(epoch):
        net.train(labels, list(data_set), rate, mini_batch)
        print('after epoch %d loss: %f' % ((i + 1),net.loss(labels[-1], net.predict(data_set[-1]))))
        rate /= 2

    ann_fcl.correct_ratio(net)