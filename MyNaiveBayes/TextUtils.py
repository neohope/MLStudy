#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
工具类
分词及集合类型转换
"""

import re
import jieba.posseg as postag


def text_parse_en(big_str_en):
    """
    英文单词分词
    """
    token_list = re.split(r'\W+', big_str_en)
    if len(token_list) == 0:
        print(token_list)
    return [tok.lower() for tok in token_list if len(tok) > 2]


def text_parse_cn(big_str_cn):
    """
    中文单词分词
    """
    words = postag.cut(big_str_cn)
    wlist=[]
    for w in words:
        wlist.append(w.word)

    return wlist


def set_of_words2vec(vocab_list, input_set):
    """
    句子转换为向量，如果字典中有这个词则将该单词置1，否则置0
    :param vocab_list: 所有单词集合列表
    :param input_set: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """

    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)

    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            # print('the word: {} is not in my vocabulary'.format(word))
            pass
    return result


def bag_words2vec(vocab_list, input_set):
    """
    句子转换为向量，每出现一次则加1
    """
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
        else:
            #print('the word: {} is not in my vocabulary'.format(word))
            pass
    return result


def create_vocab_list(data_set):
    """
    单词list转set
    """
    vocab_set = set()
    for item in data_set:
        vocab_set = vocab_set | set(item)
    return list(vocab_set)

