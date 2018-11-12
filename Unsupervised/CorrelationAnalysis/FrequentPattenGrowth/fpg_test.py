#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Frequent Pattern Growth查找频繁项集
"""

from Unsupervised.CorrelationAnalysis.FrequentPattenGrowth import fpg_utils


def load_data():
    """
    生成测试数据
    """
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


if __name__ == "__main__":
    # 加载数据
    simpDat = load_data()

    # 初始化数据集
    initSet = fpg_utils.createInitSet(simpDat)
    print('initSet: \n', initSet)

    # 创建FP树
    # 输入：dist{行：出现次数}的样本数据  和  最小的支持度
    # 输出：最终的PF-tree，通过循环获取第一层的节点，然后每一层的节点进行递归的获取每一行的字节点，也就是分支。然后所谓的指针，就是后来的指向已存在的
    myFPtree, myHeaderTab = fpg_utils.createTree(initSet, 3)
    myFPtree.disp()

    # 查询树节点的频繁子项
    print('x: \n', fpg_utils.findPrefixPath('x', myHeaderTab['x'][1]))
    print('z: \n', fpg_utils.findPrefixPath('z', myHeaderTab['z'][1]))
    print('r: \n', fpg_utils.findPrefixPath('r', myHeaderTab['r'][1]))

    # 频繁项集
    freqItemList = []
    fpg_utils.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList)
    print("freqItemList: \n", freqItemList)