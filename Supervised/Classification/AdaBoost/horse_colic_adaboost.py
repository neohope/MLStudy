#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ensemble method->boosting->adaBoost/adaptive boosting
马的疝气病
"""

import numpy as np
from Supervised.Classification.AdaBoost import adaboost_utils


def load_data_set(file_name):
    """
    加载马的疝气病的数据
    :param file_name: 文件名
    :return: 必须要是np.array或者np.matrix
    """
    num_feat = len(open(file_name).readline().split('\t'))
    data_arr = []
    label_arr = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr


def test():
    # 加载数据
    data_mat, class_labels = load_data_set('../../../Data/AdaBoost/horseColicTraining2.txt')
    data_arr_test, label_arr_test = load_data_set("../../../Data/AdaBoost/horseColicTest2.txt")

    # 训练数据
    weak_class_arr, agg_class_est = adaboost_utils.ada_boost_train_ds(data_mat, class_labels, 40)

    # ROC曲线
    adaboost_utils.plot_roc(agg_class_est, class_labels)

    # 预测
    predicting10 = adaboost_utils.ada_classify(data_arr_test, weak_class_arr)

    # 计算总样本数，错误样本数，错误率
    m = np.shape(data_arr_test)[0]
    err_arr = np.mat(np.ones((m, 1)))
    print(m)
    print(err_arr[predicting10 != np.mat(label_arr_test).T].sum())
    print(err_arr[predicting10 != np.mat(label_arr_test).T].sum()/m)

if __name__ == '__main__':
    test()
