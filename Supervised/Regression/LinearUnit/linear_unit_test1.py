#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
线性单元实现
"""

from Supervised.Regression.LinearUnit import linear_unit


def get_training_dataset():
    """
    Desc:
        构建一个简单的训练数据集
    Args:
        None
    Returns:
        input_vecs —— 训练数据集的特征部分
        labels —— 训练数据集的数据对应的标签，是一一对应的
    """
    # 构建数据集，输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，也就是输入向量的对应的标签，与工作年限对应的收入年薪
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    """
    Desc:
        使用训练数据集对线性单元进行训练
    Args:
        None
    Returns:
        lu —— 返回训练好的线性单元
    """
    # 创建感知器对象，输入参数的个数也就是特征数为 1（工作年限）
    lu = linear_unit.LinearUnit(1)
    # 获取构建的数据集
    input_vecs, labels = get_training_dataset()
    # 训练感知器，迭代 10 轮，学习率为 0.01
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return input_vecs, labels, lu


if __name__ == '__main__':
    # 首先训练线性单元
    input_vecs, labels, lu = train_linear_unit()

    # 打印训练获得的权重 和 bias
    print(lu)
    print('Work 3.4 years, monthly salary = %.2f' % lu.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % lu.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % lu.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % lu.predict([6.3]))

    linear_unit.plot(lu, input_vecs, labels)
