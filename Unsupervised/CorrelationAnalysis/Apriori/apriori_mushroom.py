#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Apriori关联分析
毒蘑菇的相似特性
"""

from Unsupervised.CorrelationAnalysis.Apriori import apriori_utils


if __name__ == "__main__":
    #

    # 得到全集的数据
    dataSet = [line.split() for line in open("../../../Data/Apriori/mushroom.txt").readlines()]
    L, supportData = apriori_utils.apriori(dataSet, minSupport=0.3)

    # 2表示毒蘑菇，1表示可食用的蘑菇
    # 如果一种蘑菇是毒蘑菇，那么它的频繁集项里的蘑菇也可很可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[2]:
        if item.intersection('2'):
            print(item)

    for item in L[3]:
        if item.intersection('2'):
            print(item)

    for item in L[4]:
        if item.intersection('2'):
            print(item)