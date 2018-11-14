#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
神经网络持久化
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def load_data():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
    return x,y


def create_net(x, y):
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(1,3,1)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    return net1


def save(net1):
    # save entire net
    torch.save(net1, 'output/net.pkl')


def save_params(net1):
    # save only the parameters
    torch.save(net1.state_dict(), 'output/net_params.pkl')


def restore_net(x, y):
    # restore entire net1 to net2
    net2 = torch.load('output/net.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(1,3,2)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params(x, y):
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('output/net_params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(1,3,3)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


if __name__ == '__main__':

    # 结果可再现
    torch.manual_seed(1)

    # 加载数据
    x, y = load_data()

    # 创建net
    net1 = create_net(x, y)

    # 保存net
    save(net1)

    # 加载net
    restore_net(x, y)

    # 保存net参数
    save_params(net1)

    # 加载net参数
    restore_params(x, y)
