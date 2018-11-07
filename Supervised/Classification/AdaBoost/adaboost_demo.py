#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ensemble method->boosting->adaBoost/adaptive boosting
AdaBoost示例
"""

import numpy as np
from Supervised.Classification.AdaBoost import adaboost_utils


def load_sim_data():
    """
    测试数据，
    :return: data_arr   feature对应的数据集
            label_arr   feature对应的分类标签
    """
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def test():
    data_mat, class_labels = load_sim_data()
    D = np.mat(np.ones((5, 1))/5)
    result = adaboost_utils.build_stump(data_mat, class_labels, D)
    print(result)

    classifier_array, agg_class_est = adaboost_utils.ada_boost_train_ds(data_mat, class_labels, 9)
    print(classifier_array, agg_class_est)


if __name__ == '__main__':
    test()
