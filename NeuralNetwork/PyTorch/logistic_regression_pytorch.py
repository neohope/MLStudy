#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
神经网络实现逻辑回归
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


def load_data():
    """
    创造测试数据
    """
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)

    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer
    x, y = Variable(x), Variable(y)

    return x,y


class MyNet(torch.nn.Module):
    """
    神经网络
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(MyNet, self).__init__()
        # 隐藏层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x_h_o = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        y = self.out(x_h_o)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return y


if __name__ == '__main__':
    # 加载数据
    x,y = load_data()

    # 创建神经网络
    net1 = MyNet(n_feature=2, n_hidden=10, n_output=2)

    # SGD随机最速下降法
    # lr表示学习率
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)

    # 损失函数，交叉熵损失
    loss_func = torch.nn.CrossEntropyLoss()

    # 可视化
    plt.ion()
    for t in range(100):
        out = net1(x)
        loss = loss_func(out, y)  # 计算两者的误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到net的 parameters 上

        # 每2次迭代，刷新一次
        if t % 2 == 0:
            plt.cla()
            # softmax激励函数处理后的最大概率才是预测值
            prediction = torch.max(F.softmax(out, dim=1), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()