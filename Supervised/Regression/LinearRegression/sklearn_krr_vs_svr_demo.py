#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
KRR: Kernel ridge regression 内核岭回归
SVR: Support Vector Regression 支持向量回归
比较 KRR 与 SVR
"""

import time
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge


if __name__ == '__main__':
    # 生成样本数据并增加噪音
    rng = np.random.RandomState(0)
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

    # SVR拟合
    t0 = time.time()
    train_size = 100
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
    svr.fit(X[:train_size], y[:train_size])
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)
    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    print("Support vector ratio: %.3f" % sv_ratio)

    # KRR拟合
    t0 = time.time()
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
    kr.fit(X[:train_size], y[:train_size])
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

    # SVR预测
    X_plot = np.linspace(0, 5, 100000)[:, None]
    t0 = time.time()
    y_svr = svr.predict(X_plot)
    svr_predict = time.time() - t0
    print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))

    # KRR预测
    t0 = time.time()
    y_kr = kr.predict(X_plot)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))

    # 预测结果可视化
    sv_ind = svr.best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors', zorder=2)
    plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1)
    plt.plot(X_plot, y_svr, c='r', label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.plot(X_plot, y_kr, c='g', label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()

    # 可视化训练和预测时间
    plt.figure()
    # 生成样本数据
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0]//5))
    sizes = np.logspace(1, 4, 7, dtype=np.int)
    for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,gamma=10), "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
        train_time = []
        test_time = []
        for train_test_size in sizes:
            t0 = time.time()
            estimator.fit(X[:train_test_size], y[:train_test_size])
            train_time.append(time.time() - t0)

            t0 = time.time()
            estimator.predict(X_plot[:1000])
            test_time.append(time.time() - t0)

        plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g", label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g", label="%s (test)" % name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Train size")
    plt.ylabel("Time (seconds)")
    plt.title('Execution Time')
    plt.legend(loc="best")

    # 可视化学习曲线
    plt.figure()
    svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
    kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    train_sizes_abs, train_scores_kr, test_scores_kr = learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r", label="SVR")
    plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g", label="KRR")
    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")
    plt.show()