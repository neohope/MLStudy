#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
AdaBoost/adaptive boosting工具类
"""

import matplotlib.pyplot as plt
import numpy as np


def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    """
    adaBoost训练
    :param data_arr: 特征标签集合
    :param class_labels: 分类标签集合
    :param num_it: 迭代次数
    :return: weak_class_arr  弱分类器的集合
            agg_class_est   预测的分类结果值
    """

    weak_class_arr = []
    m = np.shape(data_arr)[0]
    # 初始化 D，设置每个特征的权重值，平均分为m份
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_it):
        # 得到决策树的模型
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)

        # alpha 目的主要是计算每一个分类器实例的权重(加和就是分类结果)
        # 计算每个分类器的 alpha 权重值
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha

        # store Stump Params in Array
        weak_class_arr.append(best_stump)

        # 分类正确：乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
        # 分类错误：乘积为 -1，结果会受影响，所以也乘以 -1
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)

        # 判断正确的，就乘以-1，否则就乘以1
        # 计算e的expon次方，然后计算得到一个综合的概率的值
        # 结果发现： 判断错误的样本，D对于的样本权重值会变大。
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        agg_class_est += alpha * class_est

        # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        # 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        if error_rate == 0.0:
            break

    return weak_class_arr, agg_class_est


def build_stump(data_arr, class_labels, D):
    """
    得到决策树的模型
    :param data_arr: 特征标签集合
    :param class_labels: 分类标签集合
    :param D: 最初的特征权重值
    :return: bestStump    最优的分类器模型
            min_error     错误率
            best_class_est  训练后的结果集
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf

    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_err = D.T * err_arr
                """ 
                dim              表示 feature列
                thresh_val       表示树的分界值
                inequal          表示计算树左右颠倒的错误率的情况
                weighted_error   表示整体结果的错误率
                best_class_est    预测的最优结果 （与class_labels对应）
                """ 
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    # best_stump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少 (单个弱分类器)
    return best_stump, min_err, best_class_est


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    """
    将数据集，按照feature列的value进行二分法切分比较来赋值分类
    :param data_mat: Matrix数据集
    :param dimen: 特征的哪一个列
    :param thresh_val: 特征列要比较的值
    :param thresh_ineq:
    :return: np.array
    """
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    # thresh_ineq == 'lt'表示修改左边的值，gt表示修改右边的值
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array


def ada_classify(data_to_class, classifier_arr):
    """
    进行预测
    :param data_to_class: 数据集
    :param classifier_arr: 分类器列表
    :return: 正负一，也就是表示分类的结果
    """
    data_mat = np.mat(data_to_class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(
            data_mat, classifier_arr[i]['dim'],
            classifier_arr[i]['thresh'],
            classifier_arr[i]['ineq']
        )
        agg_class_est += classifier_arr[i]['alpha'] * class_est
    return np.sign(agg_class_est)


def plot_roc(pred_strengths, class_labels):
    """
    打印ROC曲线，并计算AUC的面积大小
    :param pred_strengths: 最终预测结果的权重值
    :param class_labels: 原始数据的分类结果集
    :return:
    """

    y_sum = 0.0
    # 对正样本的进行求和
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    # 正样本的概率
    y_step = 1 / float(num_pos_class)
    # 负样本的概率
    x_step = 1 / float(len(class_labels) - num_pos_class)
    # np.argsort函数返回的是数组值从小到大的索引值
    sorted_indicies = pred_strengths.argsort()

    # 图形化展示
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    # cursor光标值
    cur = (1.0, 1.0)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # 画线段
        # print([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y])
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)

    # 画对角的虚线线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')

    # 设置画图的范围区间
    ax.axis([0, 1, 0, 1])
    plt.show()

    """ 
    为了计算 AUC ，我们需要对多个小矩形的面积进行累加。
    这些小矩形的宽度是x_step，因此可以先对所有矩形的高度进行累加，最后再乘以x_step得到其总面积。
    所有高度的和(y_sum)随着x轴的每次移动而渐次增加。
    """ 
    print("the Area Under the Curve is: ", y_sum * x_step)

