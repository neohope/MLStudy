#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
利用决策树判断是否肥胖
"""

import numpy as np
import pydotplus
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO


def createDataSet():
    """
    数据读入
    """

    # 特征(身高 体重)
    # label(胖瘦)
    data = []
    labels = []
    with open("../Data/DecisionTree/fat/fat.txt") as ifile:
        for line in ifile:
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    x = np.array(data)

    # 标签转换为0/1
    labels = np.array(labels)
    y = np.zeros(labels.shape)
    y[labels == 'fat'] = 1

    return x, y


def predict_train(x_train, y_train, x_test, y_test):
    """
    使用信息熵作为划分标准，对决策树进行训练
    """

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)

    # 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
    print('Feature权重系数: %s' % clf.feature_importances_)

    # 测试准确率
    y_test_pre = clf.predict(x_test)
    print('准确率: %d' %np.mean(y_test_pre == y_test))

    return y_test_pre, clf

def show_precision_recall0(x, y, clf,  y_test, y_test_pre):
    """
    准确率与召回率
    """

    # 判断预测结果
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pre)


def show_precision_recall1(x, y, clf,  y_test, y_test_pre):
    """
    准确率与召回率
    """

    # 计算全量的预估结果
    y_pre = clf.predict_proba(x)[:, 1]

    # 输出预测结果
    # print(y_pre)
    # print(y)

    # label
    target_names = ['thin', 'fat']

    """
    precision 准确率
    recall 召回率
    f1-score  准确率和召回率的一个综合得分
    support 参与比较的数量
    """
    print("准确率与召回率")
    print(classification_report(y, y_pre, target_names=target_names))


def save_as_dot(clf):
    """
    输出为dot格式
    """
    with open("output/tree.dot", 'w') as f:
         tree.export_graphviz(clf, out_file=f)


def save_as_png(clf):
    """
    输出为png格式
    """
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    with open('output/tree.png', 'w+b') as f:
        f.write(graph.create_png())


def save_as_pdf(clf):
    """
    输出为pdf格式
    """
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("output/tree.pdf")


if __name__ == '__main__':

    # 加载数据集
    x, y = createDataSet()

    # 拆分训练数据与测试数据， 80%做训练 20%做测试
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 得到训练的预测结果集
    y_test_pre, clf = predict_train(x_train, y_train, x_test, y_test)

    # 展现 准确率与召回率
    show_precision_recall1(x, y, clf, y_test, y_test_pre)

    # 可视化输出
    #save_as_dot(clf)
    #save_as_pdf(clf)
    #save_as_png(clf)
