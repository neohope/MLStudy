#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ANN Fully Connected Layers
mnist数据集实现手写识别
"""

import warnings
from datetime import datetime
from NeuralNetwork.ANN import ann_fcl


# 忽略sigmoid函数位数溢出的警告
warnings.filterwarnings('ignore')


class Loader(object):
    """
    数据加载器
    """
    def __init__(self, path, count):
        """ 
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        """ 
        self.path = path
        self.count = count

    def get_file_content(self):
        """ 
        读取文件内容
        """ 
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return list(content)

    def to_int(self, byte):
        """ 
        将unsigned byte字符转换为整数
        """ 
        #return struct.unpack('B', byte)[0]
        return byte


class ImageLoader(Loader):
    """
    图像数据加载器
    """
    def get_picture(self, content, index):
        """ 
        内部函数，从文件中获取图像
        """ 
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        """ 
        内部函数，将图像转化为样本的输入向量
        """ 
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        """ 
        加载数据文件，获得全部样本的输入向量
        """ 
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


class LabelLoader(Loader):
    """
    标签数据加载器
    """
    def load(self):
        """ 
        加载数据文件，获得全部样本的标签向量
        """ 
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        """ 
        内部函数，将一个值转换为10维标签向量
        """ 
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def train_and_evaluate():
    """
    训练和评估
    """
    train_data_set, train_labels = ann_fcl.transpose(get_training_data_set())
    test_data_set, test_labels = ann_fcl.transpose(get_test_data_set())
    train_data_set =list(train_data_set)
    train_labels = list(train_labels)
    test_data_set = list(test_data_set)
    test_labels = list(test_labels)
    network = ann_fcl.Network([784, 100, 10])

    last_error_ratio = 1.0
    epoch = 0
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.01, 1)
        print('%s epoch %d finished, loss %f' % (now(), epoch, network.loss(train_labels[-1], network.predict(train_data_set[-1]))))
        if epoch % 2 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


def get_training_data_set():
    """ 
    获得训练数据集
    原文为60000的数据集，但训练速度过于缓慢，这里
    """ 
    image_loader = ImageLoader('../../Data/ANN/Minst/ubyte/train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('../../Data/ANN/Minst/ubyte/train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    """ 
    获得测试数据集
    """ 
    image_loader = ImageLoader('../../Data/ANN/ubyte/t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('../../Data/ANN/ubyte/t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def evaluate(network, test_data_set, test_labels):
    """
    评估
    """
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def get_result(vec):
    """
    获取向量最大值的下标
    """
    max_value_index = 0
    max_value = 0
    vec = list(vec)
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def now():
    """
    格式化时间字符串
    """
    return datetime.now().strftime('%c')


if __name__ == '__main__':
    train_and_evaluate()
