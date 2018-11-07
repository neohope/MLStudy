#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
ensemble method->bagging->random forest
随机森林法对声纳信号进行分类
"""

from random import seed, randrange, random


def load_data(filename):
    """
    加载数据
    """
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for featrue in line.split(','):
                str_f = featrue.strip()
                #最后一列为分类标签
                if str_f.isalpha():
                    lineArr.append(str_f)
                else:
                    lineArr.append(float(str_f))
            dataset.append(lineArr)
    return dataset


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Description:
        评估算法性能，返回模型得分
    Args:
        dataset     原始数据集
        algorithm   使用的算法
        n_folds     数据的份数
        *args       其他的参数
    Returns:
        scores      模型得分
    """

    # 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次 list 的元素是无重复的
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        # 训练集
        train_set = sum(train_set, [])
        # 测试集
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        #训练
        predicted = algorithm(train_set, test_set, *args)

        # 计算随机森林的预测结果的正确率
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def cross_validation_split(dataset, n_folds):
    """
    Description:
        将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的
    Args:
        dataset     原始数据集
        n_folds     数据集dataset分成n_flods份
    Returns:
        dataset_split    list集合，存放的是：将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy[index])
        dataset_split.append(fold)
    return dataset_split


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Description:
        评估算法性能，返回模型得分
    Args:
        train           训练数据集
        test            测试数据集
        max_depth       决策树深度不能太深，不然容易导致过拟合
        min_size        叶子节点的大小
        sample_size     训练数据集的样本比例
        n_trees         决策树的个数
        n_features      选取的特征的个数
    Returns:
        predictions     每一行的预测结果，bagging 预测最后的分类结果
    """

    trees = list()
    # n_trees 表示决策树的数量
    for i in range(n_trees):
        # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
        sample = subsample(train, sample_size)
        # 创建一个决策树
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    # 每一行的预测结果，bagging 预测最后的分类结果
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


def subsample(dataset, ratio):
    """
    Description:
        创建数据集的随机子样本
    Args:
        dataset         训练数据集
        ratio           训练数据集的样本比例
    Returns:
        sample          随机抽样的训练样本
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def build_tree(train, max_depth, min_size, n_features):
    """
    Description:
        创建一个决策树
    Args:
        train           训练数据集
        max_depth       决策树深度不能太深，不然容易导致过拟合
        min_size        叶子节点的大小
        n_features      选取的特征的个数
    Returns:
        root            返回决策树
    """
    # 返回最优列和相关的信息
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def get_split(dataset, n_features):
    """
        找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]
        分割完的数据 groups（left, right）
    """
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    # 随机取n_features 个特征
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)

    # 在 n_features 个特征中选出最优的特征索引
    for index in features:
        for row in dataset:
            # 找出最优的分类特征和特征值，计算gini系数
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)

            # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
            # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def test_split(index, value, dataset):
    """
    根据特征和特征值分割数据集
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, class_values):
    """
    gini系数
    """
    # class_values = [0, 1]
    # groups = (left, right)
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def split(node, max_depth, min_size, n_features, depth):
    """
    创建分割器，递归分类，直到分类结束
    """
    left, right = node['groups']
    del(node['groups'])

    # 左右都结束
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # 分类结束，防止过拟合
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # 左分支
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    # 右分支
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


def to_terminal(group):
    """
    最终分类标签
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def bagging_predict(trees, row):
    """
    Description:
        bagging预测
    Args:
        trees           决策树的集合
        row             测试数据集的每一行数据
    Returns:
        返回随机森林中，决策树结果出现次数做大的
    """

    # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def predict(node, row):
    """
    预测模型分类结果
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def accuracy_metric(actual, predicted):
    """
    计算精确度
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


if __name__ == '__main__':
    # 加载数据
    dataset = load_data('../../../Data/RandomForest/sonar-all-data.txt')

    n_folds = 5         # 分成5份数据，进行交叉验证
    max_depth = 20      # 决策树深度不能太深，不然容易导致过拟合
    min_size = 1        # 决策树的叶子节点最少的元素数量
    sample_size = 1.0   # 做决策树时候的样本的比例
    n_features = 15     # 调参（自己修改） #准确性与多样性之间的权衡

    # 测试不同森林大小时的结果
    #  1 74.29
    # 10 78.10
    # 20 79.05
    for n_trees in [1, 10, 20]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
