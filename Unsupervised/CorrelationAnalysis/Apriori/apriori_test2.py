#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Apriori关联规则
"""

from Unsupervised.CorrelationAnalysis.Apriori import apriori_utils


# 加载数据集
def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def testGenerateRules():
    # 加载测试数据集
    dataSet = load_data()

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData = apriori_utils.apriori(dataSet, minSupport=0.5)
    print ('L(0.5) is : ', L1)
    print ('supportData(0.5) is : ', supportData)

    # 生成关联规则
    rules = apriori_utils.generateRules(L1, supportData, minConf=0.5)
    print ('rules(0.5) are is ', rules)


if __name__ == "__main__":
    # 生成关联规则
    testGenerateRules()
