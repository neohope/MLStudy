#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
决策树持久化
"""

import pickle


def storeTree(inputTree, filename):
    """
    Desc:
        将之前训练好的决策树模型存储起来，使用 pickle 模块
    Args:
        inputTree -- 以前训练好的决策树模型
        filename -- 要存储的名称
    Returns:
        None
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    Desc:
        将之前存储的决策树模型使用 pickle 模块 还原出来
    Args:
        filename -- 之前存储决策树模型的文件名
    Returns:
        pickle.load(fr) -- 将之前存储的决策树模型还原出来
    """
    with open(filename, 'rb') as fr:
        return pickle.load(fr)

