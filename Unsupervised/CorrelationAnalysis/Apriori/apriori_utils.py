#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Apriori工具类
"""


def apriori(dataSet, minSupport=0.5):
    """
    Apriori方法找到频繁集
    首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。
    那么满足最小支持度要求的项集构成集合 L1。
    然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2
    然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度
    Args:
        dataSet 原始数据集
        minSupport 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """

    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)

    # 对每一行进行 set 转换，然后存放到集合中
    d=[]
    for row in dataSet:
        d.append(set(row))

    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(d, C1, minSupport)

    # 合并数据集
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        # 合并数据集
        Ck = aprioriGen(L[k-2], k)

        # 计算支持度
        Lk, supK = scanD(d, Ck, minSupport)

        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)

        if len(Lk) == 0:
            break
        L.append(Lk)
        k += 1
    return L, supportData


def createC1(dataSet):
    """
    创建集合 C1
    即对 dataSet 进行去重，排序，放入 list 中
    然后转换所有的元素为 frozenset
    Args:
        dataSet 原始数据集
    Returns:
        frozenset 返回一个 frozenset 格式的 list
    """

    # 去重
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    # 对数组进行 `从小到大` 的排序
    C1.sort()

    FC1 = []
    for row in C1:
        FC1.append(frozenset(row))

    return FC1


def scanD(d, ck, minSupport):
    """
    计算候选数据集CK在数据集D中的支持度，
    并返回支持度大于最小支持度 minSupport 的数据
    Args:
        D 数据集
        Ck 候选项集列表
        minSupport 最小支持度
    Returns:
        retList 支持度大于 minSupport 的集合
        supportData 候选项集支持度数据
    """

    # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8    
    ssCnt = {}
    for tid in d:
        for can in ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1

    numItems = (float(len(d)))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
        if support >= minSupport:
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
    例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}.
    以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
    仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
    Args:
        Lk 频繁项集列表
        k 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
    Returns:
        retList 元素两两合并的数据集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def generateRules(L, supportData, minConf=0.7):
    """
    生成关联规则
    Args:
        L 频繁项集列表
        supportData 频繁项集支持度的字典
        minConf 最小置信度
    Returns:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """
    bigRuleList = []
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]

            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    递归计算频繁项集的规则
    Args:
        freqSet 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的数组
        minConf 最小可信度
    """

    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # 生成 m+1 个长度的所有可能的 H 中的组合
        Hmp1 = aprioriGen(H, m+1)

        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)

        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    对两个元素的频繁项，计算可信度
    Args:
        freqSet 频繁项集中的元素，例如: frozenset([1, 3])
        H 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的空数组
        minConf 最小可信度
    Returns:
        prunedH 记录 可信度大于阈值的集合
    """
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:
        # 支持度定义: a -> b = support(a | b) / support(a)
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            # 支持度足够高
            print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
