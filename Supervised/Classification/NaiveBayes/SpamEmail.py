#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
删选垃圾邮件
"""

import random

import numpy as np
from MyNaiveBayes import TextUtils

from Supervised.Classification.NaiveBayes import ClassifNB


def spam_test():
    """
    删选垃圾邮件
    """

    doc_list = []
    class_list = []

    # 每种邮件都是25封
    for i in range(1, 26):
        # 添加垃圾邮件信息
        try:
            words = TextUtils.text_parse(open('../../../Data/NaiveBayes/email/spam/{}.txt'.format(i)).read())
        except:
            words = TextUtils.text_parse(open('../../../Data/NaiveBayes/email/spam/{}.txt'.format(i), encoding='Windows 1252').read())
        doc_list.append(words)
        class_list.append(1)

        # 添加非垃圾邮件
        try:
            words = TextUtils.text_parse(open('../../../Data/NaiveBayes/email/ham/{}.txt'.format(i)).read())
        except:
            words = TextUtils.text_parse(open('../../../Data/NaiveBayes/email/ham/{}.txt'.format(i), encoding='Windows 1252').read())
        doc_list.append(words)
        class_list.append(0)

    # 创建词汇表
    vocab_list = TextUtils.create_vocab_list(doc_list)

    # 随机抽取10封邮件做测试数据
    test_set = [int(num) for num in random.sample(range(50), 10)]

    # 利用剩余40封邮件训练模型
    training_set = list(set(range(50)) - set(test_set))
    training_mat = []
    training_class = []
    for doc_index in training_set:
        training_mat.append(TextUtils.set_of_words2vec(vocab_list, doc_list[doc_index]))
        training_class.append(class_list[doc_index])
    p0v, p1v, p_spam = ClassifNB.train_naive_bayes(
        np.array(training_mat),
        np.array(training_class)
    )

    # 测试10封邮件
    error_count = 0
    for doc_index in test_set:
        word_vec = TextUtils.set_of_words2vec(vocab_list, doc_list[doc_index])
        if ClassifNB.classify_naive_bayes(
            np.array(word_vec),
            p0v,
            p1v,
            p_spam
        ) != class_list[doc_index]:
            error_count += 1

    print('the error rate is {}'.format(
        error_count / len(test_set)
    ))


if __name__ == "__main__":
    spam_test()
