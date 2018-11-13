#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ANN神经网络Demo
"""

from NeuralNetwork.ANN import ann_bp


def gradient_check_test():
    """ 
    Desc:
        梯度检查测试
    Args:
        None
    Returns:
        None
    """ 
    # 创建一个有 3 层的网络，每层有 2 个节点
    net = ann_bp.Network([2, 2, 2])
    # 样本的特征
    sample_feature = [0.9, 0.1]
    # 样本对应的标签
    sample_label = [0.9, 0.1]
    # 使用梯度检查来查看是否正确
    ann_bp.gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    gradient_check_test()
