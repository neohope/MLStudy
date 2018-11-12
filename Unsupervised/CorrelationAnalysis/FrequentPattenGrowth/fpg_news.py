#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Frequent Pattern Growth查找频繁项集
新闻网站点击数据进行关联分析
"""

from Unsupervised.CorrelationAnalysis.FrequentPattenGrowth import fpg_utils

if __name__ == "__main__":
    # 加载数据并初始化
    parsedDat = [line.split() for line in open('../../../Data/FrequentPattenGrowth/kosarak.txt').readlines()]
    initSet = fpg_utils.createInitSet(parsedDat)

    # 创建FP树
    myFPtree, myHeaderTab = fpg_utils.createTree(initSet, 100000)

    # 创建条件FP树
    myFreList = []
    preFix = set([])
    fpg_utils.mineTree(myFPtree, myHeaderTab, 100000, preFix, myFreList)
    print(myFreList)
