#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
线性单元Demo
"""

from Supervised.Regression.LinearUnit import linear_unit


def get_training_dataset():
    """
    构建一个简单的训练数据集
    """
    # 构建训练数据
    # 输入向量列表，每一项的第一个是工作年限，第二个是级别
    input_vecs = [[5, 1], [3, 7], [8, 2], [1.5, 5], [10, 6]]
    # 期望的输出列表，月薪
    labels = [5200, 6700, 9300, 3500, 15500]
    return input_vecs, labels


def train_linear_unit():
    """
    使用数据训练线性单元
    """
    # 创建感知器，输入参数的特征数为2
    lu = linear_unit.LinearUnit(2)
    # 训练，迭代10轮, 学习速率为0.005
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.005)
    # 返回训练好的线性单元
    return input_vecs, labels, lu


if __name__ == '__main__':
    # 首先训练线性单元
    input_vecs, labels, lu = train_linear_unit()

    # 打印训练获得的权重 和 bias
    print(lu)
    print('Work 3.4 years, level 1, monthly salary = %.2f' % lu.predict([3.4, 1]))
    print('Work 15 years, level 2, monthly salary = %.2f' % lu.predict([15, 2]))
    print('Work 1.5 years, level 3, monthly salary = %.2f' % lu.predict([1.5, 3]))
    print('Work 6.3 years, level 4, monthly salary = %.2f' % lu.predict([6.3, 4]))

    linear_unit.plot3d(lu, input_vecs, labels)
