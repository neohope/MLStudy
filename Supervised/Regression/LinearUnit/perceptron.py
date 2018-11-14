#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Perceptron
感知机
"""

from functools import reduce


class Perceptron(object):
    """
    Desc:
        感知器类
    Args:
        None
    Returns:
        None
    """
    def __init__(self,input_num,activator):
        """
        Desc:
          初始化感知器
        Args:
          input_num —— 输入参数的个数
          activator —— 激活函数
        Returns:
          None
        """
        # 设置的激活函数
        self.activator = activator
        # 权重向量初始化为 0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为 0
        self.bias = 0.0

    def __str__(self):
        """
        Desc:
            将感知器信息打印出来
        Args:
            None
        Returns:
            None
        """
        return  'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self,input_vec):
        """
        Desc:
            输入向量，输出感知器的计算结果
        Args:
            input_vec —— 输入向量
        Returns:
            感知器的计算结果
        """
        # 将输入向量的计算结果返回
        # 调用激活函数 activator ，将输入向量输入，计算感知器的结果
        # reduce() 从左到右对一个序列的项累计地应用有两个参数的函数，以此合并序列到一个单一值
        pack = zip(input_vec,self.weights)
        multi = []
        for (x,w) in pack:
            multi.append(x*w)
        activtion = reduce(add, multi)

        # tp[0]为input_vec,tp[1]为self.weights
        return self.activator(activtion + self.bias)

    def train(self,input_vecs,labels,iteration,rate):
        """
        Desc:
            输入训练数据：一组向量、与每个向量对应的 label、 训练迭代轮数、学习率
        Args:
            input_vec —— 输入向量
            labels —— 数据对应的标签
            iteration —— 训练的迭代轮数
            rate —— 学习率
        Returns:
            None
        """
        for i in range(iteration):
            self.one_iteration(input_vecs, labels, rate)

    def one_iteration(self, input_vecs, labels, rate):
        """
        Desc:
            训练过程的一次迭代过程
        Args:
            input_vecs —— 输入向量
            labels —— 数据对应的标签
            rate —— 学习率
        Returns:
            None
        """
        # zip() 接收任意多个（包括 0 个和 1个）序列作为参数，返回一个 tuple 列表
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            output = self.update_weights(input_vec, output, label, rate)

    def update_weights(self, input_vecs, output, labels, rate):
        """
        Desc:
            按照感知器规则更新权重
        Args:
            input_vec —— 输入向量
            output —— 经过感知器规则计算得到的输出
            label —— 输入向量对应的标签
            rate —— 学习率
        Returns:
            None
        """
        # 利用感知器规则更新权重
        delta = labels -output
        # map() 接收一个函数 f 和一个 list ，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 返回
        pack  = zip(input_vecs,self.weights)
        tmp = []
        for (x,w) in pack:
            tmp.append(w+x*delta*rate)
        self.weights = tmp
        # 更新 bias
        self.bias = self.bias + delta*rate


def add(x,y):
    """
    Desc:
        reduce函数输入
    """
    return  x+y


def f10(x):
    """
    Desc:
        定义激活函数 f
    Args:
        x —— 输入向量
    Returns:
        （实现阶跃函数）大于 0 返回 1，否则返回 0
    """
    if x>0:
        return 1
    else:
        return 0

