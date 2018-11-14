#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
rnn回归
"""

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate


class MyRNN(nn.Module):
    """
    RNN 神经网络
    """
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension.
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        """
        前向传播
        """
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # save all predictions
        outs = []
        # calculate output for each time step
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


if __name__ == '__main__':
    # 可重现
    torch.manual_seed(1)

    # 构建网络
    rnn = MyRNN()

    # optimize all cnn parameters
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    h_state = None      # for initial hidden state

    plt.figure(1, figsize=(12, 5))
    plt.ion()
    for step in range(60):
        start, end = step * np.pi, (step+1)*np.pi   # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)    # float32 for converting torch FloatTensor
        y_np = np.cos(steps)

        # shape (batch, time_step, input_size)
        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        # rnn output
        prediction, h_state = rnn(x, h_state)

        # repack the hidden state, break the connection from last iteration
        h_state = Variable(h_state.data)

        loss = loss_func(prediction, y)         # cross entropy loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

        # plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()