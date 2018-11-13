#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ANN神经网络Demo
"""

from NeuralNetwork.ANN import ann_fcl


def gradient_check():
    """
    梯度检查
    """
    labels, data_set = ann_fcl.transpose(ann_fcl.train_data_set())
    labels = list(labels)
    data_set = list(data_set)

    net = ann_fcl.Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net


if __name__ == '__main__':
    gradient_check()
