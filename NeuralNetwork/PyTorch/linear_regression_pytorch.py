#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
神经网络实现线性回归
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyNet(torch.nn.Module):
    """
    神经网络
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(MyNet, self).__init__()
        # 中间层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 前向传播
        # 隐藏层-输出结果（输入加权、激活输出）
        x_h_o = F.relu(self.hidden(x))
        # 输出层-预测结果（输入加权）
        y = self.predict(x_h_o)
        return y


def load_data():
    """
    创造测试数据，并引入噪声
    """
    # torch.linspace 表示在 -1和1之间等距采取100各点
    # torch.unsqueeze 表示对老的tensor定位输出的方向，dim表示以行/列的形式输出
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    # 用 Variable 来修饰这些数据 tensor
    x, y = Variable(x), Variable(y)
    return x,y


if __name__ == '__main__':
    # 加载数据
    x, y = load_data()

    # 创建神经网络
    net = MyNet(n_feature=1, n_hidden=10, n_output=1)

    # SGD随机最速下降法
    # lr表示学习率
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    # 损失函数，均方误差
    loss_func = torch.nn.MSELoss()

    # 可视化
    plt.ion()
    for t in range(100):
        # 预测
        prediction = net(x)
        # 计算均方误差
        loss = loss_func(prediction, y)
        # 清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.backward()
        # 将参数更新值施加到 net 的 parameters 上
        optimizer.step()

        # 每5次迭代，刷新一次
        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, "Loss={0}".format(loss.item()), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()