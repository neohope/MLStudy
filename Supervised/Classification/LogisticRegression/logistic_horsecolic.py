#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
逻辑回归
"""

import numpy as np
from Supervised.Classification.LogisticRegression import classify_logistic

def load_data():
    """
    加载数据集
    """
    with open('../../../Data/LogisticRegression/HorseColic/HorseColicTraining.txt', 'r') as f_train:
        trian_data=f_train.readlines()
    training_set = []
    training_labels = []
    for line in trian_data:
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue
        line_arr = [float(curr_line[i]) for i in range(21)]
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))

    with open('../../../Data/LogisticRegression/HorseColic/HorseColicTest.txt', 'r') as f_test:
        test_data=f_test.readlines()
    test_set = []
    test_labels = []
    for line in test_data:
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue
        line_arr = [float(curr_line[i]) for i in range(21)]
        test_set.append(line_arr)
        test_labels.append(float(curr_line[21]))

    return training_set,training_labels,test_set,test_labels


def colic_test(training_set, training_labels, test_set, test_labels):
    """
    训练
    """
    # 随机梯度下降算法
    train_weights = classify_logistic.stoc_grad_ascent(np.array(training_set), training_labels, 500)

    # 测试效果
    error_count = 0
    num_test_vec = 0.0
    for ts in test_set:
        if int(classify_logistic.classify_vector(np.array(ts), train_weights)) != test_labels[int(num_test_vec)]:
            error_count += 1
        num_test_vec += 1

    error_rate = error_count / num_test_vec
    print('the error rate is {}'.format(error_rate))
    return error_rate


def multi_test(training_set, training_labels, test_set, test_labels):
    """
    调用 colicTest() 10次并求结果的平均值
    错误率还是挺高的在0.34左右
    """
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += colic_test(training_set, training_labels, test_set, test_labels)
    print('after {} iteration the average error rate is {}'.format(num_tests, error_sum / num_tests))

if __name__ == '__main__':
    training_set, training_labels, test_set, test_labels = load_data()
    colic_test(training_set, training_labels, test_set, test_labels)
    multi_test(training_set, training_labels, test_set, test_labels)

