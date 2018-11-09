#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Apriori关联分析
"""

from Unsupervised.CorrelationAnalysis.Apriori import apriori_utils


# 加载数据集
def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def testApriori():
    # 加载测试数据集
    dataSet = load_data()

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori_utils.apriori(dataSet, minSupport=0.7)
    print ('L(0.7) is : ', L1)
    print ('supportData(0.7) is : ', supportData1)

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori_utils.apriori(dataSet, minSupport=0.5)
    print ('L(0.5) is : ', L2)
    print ('supportData(0.5) is : ', supportData2)


if __name__ == "__main__":
    # 测试 Apriori 算法
    testApriori()
