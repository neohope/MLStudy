#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
cnn
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Hyper Parameters
BATCH_SIZE = 50
TRAIN_TATIO = 0.8
LR = 0.0001


class CustomedDataSet(Dataset):
    def __init__(self, pd_data, data_type=True):
        self.data_type = data_type
        if self.data_type:
            trainX = pd_data
            trainY = trainX.label.values.tolist()
            trainX = trainX.drop('label', axis=1).values.reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd_data
            testX = testX.values.reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX

    def __getitem__(self, index):
        if self.data_type:
            return torch.Tensor(self.datalist[index].astype(float)), self.labellist[index]
        else:
            return torch.Tensor(self.datalist[index].astype(float))

    def __len__(self):
        return self.datalist.shape[0]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # input shape (1, 28, 28)
            # output shape (16, 28, 28)  28=(width+2*padding-kernel_size)/stride+1
            nn.Conv2d(
                in_channels=1,    # 输入信号的通道数
                out_channels=16,  # 卷积后输出结果的通道数
                kernel_size=5,    # 卷积核的形状
                stride=1,         # 卷积核得步长
                padding=2,        # 处理边界时在每个维度首尾补0数量 padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            # output shape (16, 14, 14)  14=(width+0*padding-dilation*(kernel_size-1)-1)/stride+1
            # dilation=1 默认值为0
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化操作时的窗口大小
        )
        # output shape (32, 7, 7)
        self.conv2 = nn.Sequential(      # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2, 2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 扁平化操作
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


def model_train():
    # 加载数据
    loader_train, loader_test = load_data_train()

    # 创建CNN模型
    cnn = CNN()

    # 设置优化器和损失函数
    optimizer, loss_func = optimizer_lossfunction(cnn)

    # 训练模型
    cnn = train_model(cnn, optimizer, loss_func, loader_train, loader_test)
    return cnn


def load_data_train():
    # 先转换成 torch 能识别的 Dataset
    pd_train = pd.read_csv('../../Data/PyTorch/DigitRecognizer/train.csv', header=0)

    # 找到数据分割点
    row = pd_train.shape[0]
    split_num = int(TRAIN_TATIO*row)

    # 分割数据
    pd_data_train = pd_train[:split_num]
    pd_data_test = pd_train[split_num:]
    data_train = CustomedDataSet(pd_data_train, data_type=True)
    data_test = CustomedDataSet(pd_data_test, data_type=True)

    # 数据读取器 (Data Loader)
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    loader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(dataset=data_test, batch_size=pd_data_test.shape[0], shuffle=True)
    return loader_train, loader_test


def optimizer_lossfunction(cnn):
    '''
    设置优化器和损失函数
    '''
    # optimizer 优化器 是训练的工具，lr表示学习率
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.99))  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    return optimizer, loss_func


def train_model(cnn, optimizer, loss_func, loader_train, loader_test):
    plt.ion()
    EPOCH = 2
    for epoch in range(EPOCH):
        num = 0
        # gives batch data, normalize x when iterate train_loader
        for step, (x, y) in enumerate(loader_train):
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y

            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                num += 1
                for _, (x_t, y_test) in enumerate(loader_test):
                    x_test = Variable(x_t)  # batch x
                    test_output, last_layer = cnn(x_test)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    accuracy = float(sum(pred_y == y_test)) / float(y_test.size(0))
                    print('Epoch: ', epoch, '| Num: ',  num, '| Step: ',  step, '| train loss: %.4f' % loss.item(), '| test accuracy: %.4f' % accuracy)
                    # 可视化展现
                    show(last_layer, y_test)
    plt.ioff()
    return cnn


def show(last_layer, y_test):
    try:
        from sklearn.manifold import TSNE
        HAS_SK = True
    except:
        HAS_SK = False
        print('Please install sklearn for layer visualization')

    if HAS_SK:
        # Visualization of trained flatten layer (T-SNE)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
        labels = y_test.numpy()[:plot_only]
        plot_with_labels(low_dim_embs, labels)


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        from matplotlib import cm
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


def load_data_pre():
    pd_pre = pd.read_csv('../../Data/PyTorch/DigitRecognizer/test.csv', header=0)
    pd_pre = pd_pre[:100]
    data_pre = CustomedDataSet(pd_pre, data_type=False)
    loader_pre = DataLoader(dataset=data_pre, batch_size=BATCH_SIZE, shuffle=True)
    return loader_pre


def prediction(cnn, loader_pre):
    for step, (x) in enumerate(loader_pre):
        b_x = Variable(x)
        test_output, _ = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print('prediction number: ', pred_y)

    return pred_y


if __name__ == "__main__":
    # 可重现
    torch.manual_seed(1)

    # 训练模型
    file_path = 'output/digit_recognizer.pkl'
    cnn = model_train()

    # 模型保存
    torch.save(cnn, file_path)

    # 保存，加载网络
    cnn1 = torch.load(file_path)

    # 预测数据
    loader_pre = load_data_pre()
    pre_data = prediction(cnn1, loader_pre)
