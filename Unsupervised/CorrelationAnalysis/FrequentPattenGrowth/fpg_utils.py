#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Frequent Pattern Growth 工具类
the FP-Growth algorithm needs:
1. FP-tree (class treeNode)
2. header table (use dict)
"""

class treeNode:
    """
    FP树节点
    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self._count = numOccur
        self.nodeLink = None
        # needs to be updated
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self._count += numOccur

    def disp(self, ind=1):
        """
        用于将树以文本形式显示
        """
        #print('  '*ind, self.name, ' ', self._count)
        for child in self.children.values():
            child.disp(ind+1)


def createInitSet(dataSet):
    """
    初始化数据集，返回出现频率
    """
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict


def createTree(dataSet, minSup=1):
    """
    生成FP树
    Args:
        dataSet  dist{行：出现次数}的样本数据
        minSup   最小的支持度
    Returns:
        retTree  FP-tree
        headerTable 满足minSup {所有的元素+(value, treeNode)}
    """

    # 支持度>=minSup的dist{所有元素：出现的次数}
    headerTable = {}

    # 循环 dist{行：出现次数}的样本数据
    for trans in dataSet:
        # 对所有的行进行循环，得到行里面的所有元素
        # 统计每一行中，每个元素出现的总次数
        for item in trans:
            # 例如： {'ababa': 3}  count(a)=3+3+3=9   count(b)=3+3=6
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 删除 headerTable中，元素次数<最小支持度的元素
    for k in list(headerTable.keys()):  # python3中.keys()返回的是迭代器不是list,不能在遍历时对其改变。
        if headerTable[k] < minSup:
            del(headerTable[k])

    # 满足minSup: set(各元素集合)
    freqItemSet = set(headerTable.keys())

    # 如果不存在，直接返回None
    if len(freqItemSet) == 0:
        return None, None

    for k in headerTable:
        # 格式化： dist{元素key: [元素次数, None]}
        headerTable[k] = [headerTable[k], None]

    # 树根
    retTree = treeNode('Null Set', 1, None)

    # 循环 dist{行：出现次数}的样本数据
    for tranSet, count in dataSet.items():
        # localD = dist{元素key: 元素总出现次数}
        localD = {}
        for item in tranSet:
            # 判断是否在满足minSup的集合中
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        # 对每一行的key 进行排序，然后开始往树添加枝丫，直到丰满
        # 第二次，如果在同一个排名下出现，那么就对该枝丫的值进行追加，继续递归调用！
        if len(localD) > 0:
            # p=key,value; 所以是通过value值的大小，进行从大到小进行排序
            # orderedItems 表示取出元组的key值，也就是字母本身，但是字母本身是大到小的顺序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]

            # 填充树，通过有序的orderedItems的第一位，进行顺序填充 第一层的子节点。
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    """
    更新FP-tree，第二次遍历
    Args:
        items       满足minSup 排序后的元素key的数组（大到小的排序）
        inTree      空的Tree对象
        headerTable 满足minSup {所有的元素+(value, treeNode)}
        count       原数据集中每一组Kay出现的次数
    """

    # 针对每一行的数据最大的key
    # 如果该元素在 inTree.children 这个字典中，就进行累加
    # 如果该元素不存在 就 inTree.children 字典中新增key，value为初始化的 treeNode 对象
    if items[0] in inTree.children:
        # 更新 最大元素，对应的 treeNode 对象的count进行叠加
        inTree.children[items[0]].inc(count)
    else:
        # 如果不存在子节点，我们为该inTree添加子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)

        # 如果满足minSup的dist字典的value值第二位为null， 我们就设置该元素为 本节点对应的tree节点
        # 如果元素第二位不为null，我们就更新header节点
        if headerTable[items[0]][1] is None:
            # headerTable只记录第一次节点出现的位置
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 本质上是修改headerTable的key对应的Tree，的nodeLink值
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        # 递归的调用，在items[0]的基础上，添加item0[1]做子节点， count只要循环的进行累计加和而已，统计出节点的最后的统计值。
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    """
    更新头指针，建立相同元素之间的关系，
    例如： 左边的r指向右边的r值，就是后出现的相同元素 指向 已经出现的元素。
    从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    性能：如果链表很长可能会遇到迭代调用的次数限制。

    Args:
        nodeToTest  满足minSup {所有的元素+(value, treeNode)}
        targetNode  Tree对象的子节点
    """
    # 建立相同元素之间的关系，例如： 左边的r指向右边的r值
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    创建条件FP树
    Args:
        inTree       myFPtree
        headerTable  满足minSup {所有的元素+(value, treeNode)}
        minSup       最小支持项集
        preFix       preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        freqItemList 用来存储频繁子项的列表
    """
    # 通过value进行从小到大的排序， 得到频繁项集的key
    # 最小支持项集的key的list集合
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]

    # 循环遍历 最频繁项集的key，从小到大的递归寻找对应的频繁项集
    for basePat in bigL:
        # preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])

        # 构建FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)

        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if myHead is not None:
            myCondTree.disp(1)
            # 递归 myHead 找出频繁项集
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def findPrefixPath(basePat, treeNode):
    """
    基础数据集

    Args:
        basePat  要查询的节点值
        treeNode 查询的节点所在的当前nodeTree
    Returns:
        condPats 对非basePat的倒叙值作为key,赋值为count数
    """
    condPats = {}
    # 对 treeNode的link进行循环
    while treeNode is not None:
        prefixPath = []

        # 寻找改节点的父节点，相当于找到了该节点的频繁项集
        ascendTree(treeNode, prefixPath)

        # 排除自身这个元素，判断是否存在父元素（所以要>1, 说明存在父元素）
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            # prefixPath[1:] 变frozenset后，字母就变无序了
            # condPats[frozenset(prefixPath)] = treeNode._count
            condPats[frozenset(prefixPath[1:])] = treeNode._count

        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
    return condPats


def ascendTree(leafNode, prefixPath):
    """
    如果存在父节点，就记录当前节点的name值
    Args:
        leafNode   查询的节点对于的nodeTree
        prefixPath 要查询的节点值
    """
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
