#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Tree递归神经网络实现
"""

import numpy as np


class TreeNode(object):
    """
    递归神经网络节点
    """
    def __init__(self, data, children=[], children_data=[]):
        self.parent = None
        self.children = children
        self.children_data = children_data
        self.data = data
        for child in children:
            child.parent = self


class RecursiveLayer(object):
    """
    Tree递归神经网络
    """
    def __init__(self, node_width, child_count, activator, learning_rate):
        """ 
        递归神经网络构造函数
        node_width: 表示每个节点的向量的维度
        child_count: 每个父节点有几个子节点
        activator: 激活函数对象
        learning_rate: 梯度下降算法学习率
        """ 
        self.node_width = node_width
        self.child_count = child_count
        self.activator = activator
        self.learning_rate = learning_rate
        # 权重数组W
        self.W = np.random.uniform(-1e-4, 1e-4, (node_width, node_width * child_count))
        # 偏置项b
        self.b = np.zeros((node_width, 1))
        # 递归神经网络生成的树的根节点
        self.root = None

    def forward(self, *children):
        """ 
        前向传播
        """ 
        children_data = self.concatenate(children)
        parent_data = self.activator.forward(np.dot(self.W, children_data) + self.b)
        self.root = TreeNode(parent_data, children , children_data)

    def backward(self, parent_delta):
        """ 
        BPTS反向传播
        """ 
        self.calc_delta(parent_delta, self.root)
        self.W_grad, self.b_grad = self.calc_gradient(self.root)

    def update(self):
        """ 
        使用SGD算法更新权重
        """ 
        self.W -= self.learning_rate * self.W_grad
        self.b -= self.learning_rate * self.b_grad

    def reset_state(self):
        """ 
        状态重置
        """ 
        self.root = None

    def concatenate(self, tree_nodes):
        """ 
        将各个树节点中的数据拼接成一个长向量
        """ 
        concat = np.zeros((0,1))
        for node in tree_nodes:
            concat = np.concatenate((concat, node.data))
        return concat

    def calc_delta(self, parent_delta, parent):
        """ 
        计算每个节点的delta
        """ 
        parent.delta = parent_delta
        if parent.children:
            # 根据式2计算每个子节点的delta
            children_delta = np.dot(self.W.T, parent_delta) * (self.activator.backward(parent.children_data))

            # slices = [(子节点编号，子节点delta起始位置，子节点delta结束位置)]
            slices = [(i, i * self.node_width, 
                        (i + 1) * self.node_width)
                        for i in range(self.child_count)]
            # 针对每个子节点，递归调用calc_delta函数
            for s in slices:
                self.calc_delta(children_delta[s[1]:s[2]], parent.children[s[0]])

    def calc_gradient(self, parent):
        """ 
        计算每个节点权重的梯度，并将它们求和，得到最终的梯度
        """ 
        W_grad = np.zeros((self.node_width, self.node_width * self.child_count))
        b_grad = np.zeros((self.node_width, 1))
        if not parent.children:
            return W_grad, b_grad

        parent.W_grad = np.dot(parent.delta, parent.children_data.T)
        parent.b_grad = parent.delta
        W_grad += parent.W_grad
        b_grad += parent.b_grad
        for child in parent.children:
            W, b = self.calc_gradient(child)
            W_grad += W
            b_grad += b
        return W_grad, b_grad

    def dump(self, **kwArgs):
        print('root.ubyte: %s' % self.root.data)
        print('root.children_data: %s' % self.root.children_data)
        if 'dump_grad'in kwArgs:
            print('W_grad: %s' % self.W_grad)
            print('b_grad: %s' % self.b_grad)


def data_set():
    children = [
        TreeNode(np.array([[1],[2]])),
        TreeNode(np.array([[3],[4]])),
        TreeNode(np.array([[5],[6]]))
    ]
    d = np.array([[0.5],[0.8]])
    return children, d
