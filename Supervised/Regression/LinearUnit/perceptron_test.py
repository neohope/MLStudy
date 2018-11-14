#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
线性单元实现
"""

from Supervised.Regression.LinearUnit import perceptron


def get_training_dataset():
    """
    Desc:
        基于 and 真值表来构建/获取训练数据集
    Args:
        None
    Returns:
        input_vecs —— 输入向量
        labels —— 输入向量对应的标签
    """
    # 构建训练数据，输入向量的列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，也就是上面的输入向量的列表中数据对应的标签，是一一对应的

    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perception():
    """
    Desc:
        使用 and 真值表来训练感知器
    Args:
        None
    Returns:
        p —— 返回训练好的感知器
    """
    # 载入数据
    input_vecs, labels = get_training_dataset()

    # 创建感知器，输入参数的个数是 2 个（因为 and 是个二元函数），激活函数为 f
    p = perceptron.Perceptron(2, perceptron.f10)

    # 进行训练，迭代 10 轮，学习速率为 0.1
    p.train(input_vecs, labels, 10, 0.1)

    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练 and 感知器
    and_perceptron = train_and_perception()
    print(and_perceptron)

    # 测试
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))