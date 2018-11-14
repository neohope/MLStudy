#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ANN神经网络Demo
"""

import numpy as np
from NeuralNetwork.ANN import ann_bp


def train_data_set():
    """
    Desc:
        获取训练数据集
    Args:
        None
    Returns:
        labels --- 训练数据集每条数据对应的标签
    """
    # 调用 Normalizer() 类
    normalizer = ann_bp.Normalizer()
    # 初始化一个 list，用来存储后面的数据
    data_set = []
    labels = []
    # 0 到 256 ，其中以 8 为步长
    for i in range(0, 256, 8):
        # 调用 normalizer 对象的 norm 方法
        n = normalizer.norm(int(np.random.uniform(0, 256)))
        # 在 data_set 中 append n
        data_set.append(n)
        # 在 labels 中 append n
        labels.append(n)
    # 将它们返回
    return labels, data_set


def test(net,data):
    """ 
    Desc:
        对全连接神经网络进行测试
    Args:
        network --- 神经网络对象
        ubyte ------ 测试数据集
    Returns:
        None
    """ 
    # 调用 Normalizer() 类
    normalizer = ann_bp.Normalizer()
    # 调用 norm 方法，对数据进行规范化
    norm_data = normalizer.norm(data)
    norm_data = list(norm_data)
    # 对测试数据进行预测
    predict_data = net.predict(norm_data)
    # 将结果打印出来
    print('testdata:\n{0}\npredict:\n{1}'.format(data, normalizer.denorm(predict_data)))


if __name__ == '__main__':
    # 初始化一个神经网络，输入层 8 个节点，隐藏层 3 个节点，输出层 8 个节点
    net = ann_bp.Network([8, 3, 8])

    # 加载数据
    labels, data_set = train_data_set()

    test(net,data_set)
