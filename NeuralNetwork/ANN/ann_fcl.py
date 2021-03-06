#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ANN神经网络实现
Fully Connected Layers
"""

from functools import reduce
import numpy as np
from NeuralNetwork.ANN.activators import SigmoidActivator


class Network(object):
    def __init__(self, layers):
        """ 
        神经网络类
        """ 
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1],SigmoidActivator()))

    def predict(self, sample):
        """ 
        使用神经网络实现预测
        sample: 输入样本
        """ 
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """ 
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """ 
        for i in range(epoch):
            for d in range(len(list(data_set))):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        """
        训练一个样本
        """
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        """
        计算梯度
        """
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        """
        更新权重
        """
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        """
        输出
        """
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        """
        损失函数
        """
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        """ 
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        """ 

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i,j]))


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        全连接层
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,(output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向传播
        input_array: 输入向量，维度必须等于input_size
        """
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        后向传播
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        """
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        """
        使用梯度下降算法更新权重
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        """
        输出
        """
        print('W: %s\nb:%s' % (self.W, self.b))


class Normalizer(object):
    """
    Desc:
        归一化工具类
    """
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)


def correct_ratio(network):
    """
    Desc:
        计算神经网络的正确率
    Args:
        network --- 神经网络对象
    Returns:
        None
    """
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


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
    normalizer = Normalizer()
    # 初始化一个 list，用来存储后面的数据
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )




