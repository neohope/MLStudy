#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
神经网络
Artificial Neural Networks
math
"""

import numpy as np
import matplotlib.pyplot as plt


def initialize_network(X):
    """
    初始化神经网络
    这里其实做的很简单，每个神经元仅仅记录了上一层的系数
    """
    input_neurons = len(X[0])
    hidden_neurons = input_neurons + 1
    output_neurons = 2
    n_hidden_layers = 1

    # 增加多个隐藏层
    # 每一层参数个数，是上一层的大小
    net = list()
    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])
        hidden_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
        net.append(hidden_layer)

    # 增加输出层
    # 每一层参数个数，是上一层的大小
    output_layer = [{'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)

    return net


def training(net, X, epochs, lrate, n_outputs):
    """
    训练神经网络
    """
    errors = []
    for epoch in range(epochs):
        sum_error = 0

        # 预测输出
        # 计算误差
        # 修正系数
        for i, row in enumerate(X):
            outputs = forward_propagation(net, row)

            expected = [0.0 for i in range(n_outputs)]
            expected[y[i]] = 1

            sum_error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
            back_propagation(net, row, expected)
            update_weights(net, row, 0.05)

        #每10000次输出一次结果
        if epoch % 10000 == 0:
            print('>epoch=%d,error=%.3f' % (epoch, sum_error))
            errors.append(sum_error)

    return errors


def forward_propagation(net, input):
    """
    正向传播
    """
    row = input
    for layer in net:
        prev_input = np.array([])
        for neuron in layer:
            sum = neuron['weights'].T.dot(row)
            result = activate_sigmoid(sum)
            neuron['result'] = result
            prev_input = np.append(prev_input, [result])
        row = prev_input

    return row


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))


def back_propagation(net, row, expected):
    """
    反向传播
    """
    for i in reversed(range(len(net))):
        layer = net[i]
        errors = np.array([])

        if i == len(net) - 1:
            results = [neuron['result'] for neuron in layer]
            errors = expected - np.array(results)
        else:
            for j in range(len(layer)):
                herror = 0
                nextlayer = net[i + 1]
                for neuron in nextlayer:
                    herror += (neuron['weights'][j] * neuron['delta'])
                errors = np.append(errors, [herror])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['result'])


def sigmoid_derivative(output):
    return output*(1.0-output)


def update_weights(net, input, lrate):
    """
    调整权重系数
    """
    for i in range(len(net)):
        inputs = input
        if i != 0:
            inputs = [neuron['result'] for neuron in net[i - 1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]


def predict(network, row):
    """
    预测
    """
    outputs = forward_propagation(net, row)
    return outputs


def print_network(net):
    """
    输出神经网络
    """
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


if __name__ == '__main__':
    #创建数据
    XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    X=XORdata[:,0:2]
    y=XORdata[:,-1]

    #创建神经网络，并初始化系数
    net=initialize_network(X)
    print_network(net)

    # 训练神经网络
    errors=training(net, X, 100000, 0.05, 2)
    print_network(net)

    # 展示训练时错误下降情况
    epochs=[0,1,2,3,4,5,6,7,8,9]
    plt.plot(epochs,errors)
    plt.xlabel("epochs in 10000's")
    plt.ylabel('error')
    plt.show()

    # 预测输出
    pred=predict(net,np.array([1,0]))
    output=np.argmax(pred)
    print(output)
