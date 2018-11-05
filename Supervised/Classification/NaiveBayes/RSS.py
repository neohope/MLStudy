#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
查找高频词汇
"""

import random
from operator import itemgetter
import feedparser
import numpy as np
from Supervised.Classification.NaiveBayes import TextUtils
from Supervised.Classification.NaiveBayes import ClassifNB


def calc_most_freq(vocab_list, full_text):
    # 去除高频词
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=itemgetter(1), reverse=True)
    return sorted_freq[0:10]


def local_words(feed1, feed0):
    """
    判断RSS来源
    """

    # 加载rss数据
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed0), len(feed1))
    for i in range(min_len):
        word_list = TextUtils.text_parse_cn(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = TextUtils.text_parse_cn(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = TextUtils.create_vocab_list(doc_list)

    # 去掉高频词
    top30words = calc_most_freq(vocab_list, full_text)
    for pair in top30words:
        if pair[0] in vocab_list:
            vocab_list.remove(pair[0])

    # 随机抽取3条数据做测试数据
    test_set = [int(num) for num in random.sample(range(2 * min_len), 3)]
    training_set = list(set(range(2 * min_len)) - set(test_set))

    # 利用剩余数据训练模型
    training_mat = []
    training_class = []
    for doc_index in training_set:
        training_mat.append(TextUtils.bag_words2vec(vocab_list, doc_list[doc_index]))
        training_class.append(class_list[doc_index])
    p0v, p1v, p_spam = ClassifNB.train_naive_bayes(
        np.array(training_mat),
        np.array(training_class)
    )

    # 测试3条数据
    error_count = 0
    for doc_index in test_set:
        word_vec = TextUtils.bag_words2vec(vocab_list, doc_list[doc_index])
        if ClassifNB.classify_naive_bayes(
            np.array(word_vec),
            p0v,
            p1v,
            p_spam
        ) != class_list[doc_index]:
            error_count += 1
    print("the error rate is {}".format(error_count / len(test_set)))
    return vocab_list, p0v, p1v


def test_rss():
    """
    判断RSS来源
    数据量太小了
    """
    ny = feedparser.parse('http://rss.sina.com.cn/roll/sports/hot_roll.xml')
    sf = feedparser.parse('http://rss.sina.com.cn/tech/rollnews.xml')
    vocab_list, p_sf, p_nf = local_words(ny, sf)


def get_top_words():
    """
    判断RSS来源，输出各来源的关键字
    数据量太小了
    """
    ny = feedparser.parse('http://rss.sina.com.cn/roll/sports/hot_roll.xml')
    sf = feedparser.parse('http://rss.sina.com.cn/tech/rollnews.xml')
    vocab_list, p_sf, p_ny = local_words(ny, sf)

    top_ny = []
    top_sf = []
    for i in range(len(p_sf)):
        if p_sf[i] > -6.0:
            top_sf.append((vocab_list[i], p_sf[i]))
        if p_ny[i] > -6.0:
            top_ny.append((vocab_list[i], p_ny[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)

    print('this is sports:')
    for item in sorted_sf:
        print(item[0])

    print('his is tech:')
    for item in sorted_ny:
        print(item[0])

if __name__ == "__main__":
    #test_rss()
    get_top_words()
