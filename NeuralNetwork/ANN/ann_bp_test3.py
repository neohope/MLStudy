#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ANN神经网络Demo
"""

from NeuralNetwork.ANN import ann_bp
from NeuralNetwork.ANN import ann_bp_test1


def train(network):
    """
    Desc:
        使用神经网络进行训练
    Args:
        network --- 神经网络对象
    Returns:
        None
    """
    # 获取训练数据集
    labels, data_set = ann_bp_test1.train_data_set()
    labels = list(labels)
    data_set = list(labels)
    # 调用 network 中的 train方法来训练神经网络
    network.train(labels, data_set, 0.3, 50)


def test():
    # 初始化一个神经网络，输入层 8 个节点，隐藏层 3 个节点，输出层 8 个节点
    net = ann_bp.Network([8, 3, 8])
    # 训练神经网络
    train(net)
    # 将神经网络的信息打印出来
    net.dump()
    # 打印出神经网络的正确率
    ann_bp.correct_ratio(net)


if __name__ == '__main__':
    test()
