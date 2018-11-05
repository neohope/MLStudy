#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
L1与L2区别
随着C的变小L1的惩罚越来越大
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def test():
    # 加载数据
    digits = datasets.load_digits()
    X, Y = digits.data, digits.target
    X = StandardScaler().fit_transform(X)

    # 数据划分为0,1
    Y = (Y>4).astype(np.int)

    # 设置正则化参数
    for i, C in enumerate((100, 1, 0.01)):
        print("C=%.2f" % C)
        plt.text(-8, 3, "C = %.2f" % C)

        # l1
        # coef_l1_LR contains zeros
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
        clf_l1_LR.fit(X, Y)
        coef_l1_LR = clf_l1_LR.coef_.ravel()
        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("Score with L1 penalty: %.4f" % clf_l1_LR.score(X, Y))

        l1_plot = plt.subplot(3, 2, 2 * i + 1)
        if i == 0:
            l1_plot.set_title("L1 penalty")
        l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest', cmap='binary', vmax=1, vmin=0)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())

        # l2
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
        clf_l2_LR.fit(X, Y)
        coef_l2_LR = clf_l2_LR.coef_.ravel()
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, Y))


        l2_plot = plt.subplot(3, 2, 2 * (i + 1))
        if i == 0:
            l2_plot.set_title("L2 penalty")
        l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest', cmap='binary', vmax=1, vmin=0)
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())

    plt.show()


if __name__ == '__main__':
    test()

