#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
屏蔽留言板的侮辱性言论
"""

import numpy as np
from MyNaiveBayes import ClassifNB
from MyNaiveBayes import TextUtils


def load_data_set():
    """
    创建数据集
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def testing_naive_bayes():
    """
    测试朴素贝叶斯算法
    """

    # 加载数据集
    list_post, list_classes = load_data_set()

    # 创建单词集合
    vocab_list = TextUtils.create_vocab_list(list_post)

    # 计算单词是否出现并创建数据矩阵
    train_mat = []
    for post_in in list_post:
        train_mat.append(
            TextUtils.set_of_words2vec(vocab_list, post_in)
        )

    # 训练数据
    p0v, p1v, p_abusive = ClassifNB.train_naive_bayes(np.array(train_mat), np.array(list_classes))

    # 测试数据
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(TextUtils.set_of_words2vec(vocab_list, test_one))
    print('the result is: {}'.format(ClassifNB.classify_naive_bayes(test_one_doc, p0v, p1v, p_abusive)))

    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(TextUtils.set_of_words2vec(vocab_list, test_two))
    print('the result is: {}'.format(ClassifNB.classify_naive_bayes(test_two_doc, p0v, p1v, p_abusive)))


if __name__ == "__main__":
    testing_naive_bayes()
