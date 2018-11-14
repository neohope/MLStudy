#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
rnn分类
基于pytorch的LSTM
"""

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64         # rnn batch size
TIME_STEP = 28          # rnn time step, image height
INPUT_SIZE = 28         # rnn input size, image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # 前向传播
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def load_data(show_one):
    # 加载Mnist digital数据集
    train_data = dsets.MNIST(
        root='../../Data/PyTorch/Minst/',
        train=True,
        # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST,  # download it if you don't have it
    )

    if show_one:
        show_one_data(train_data)

    return train_data


def show_one_data(train_data):
    # 展示一个图形
    print(train_data.train_data.size())  # (60000, 28, 28)
    print(train_data.train_labels.size())  # (60000)
    plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.train_labels[0])
    plt.show()


if __name__ == '__main__':
    # 可重现
    torch.manual_seed(1)

    # 加载数据集
    # 并选择2000条数据进行处理
    train_data=load_data(False)
    test_data = dsets.MNIST(root='../../Data/PyTorch/Minst/', train=False, transform=transforms.ToTensor())
    # shape (2000, 28, 28) value in range(0,1)
    test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels.numpy().squeeze()[:2000]

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # RNN
    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # 训练
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # gives batch data
            b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
            b_y = Variable(y)  # batch y

            output = rnn(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                test_output = rnn(test_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.4f' % accuracy)

    # 输出10个预测数据
    test_output = rnn(test_x[:100].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:100], 'real number')
