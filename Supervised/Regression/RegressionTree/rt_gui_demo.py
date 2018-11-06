#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
树回归
"""

from Supervised.Regression.RegressionTree import regress_tree_utils
import tkinter as tk
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def drawNewTree():
    """
    确认按钮的消息函数
    """
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)


def getInputs():
    """
    从界面获取输入参数
    """
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, tk.END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, tk.END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def reDraw(tolS, tolN):
    """
    从界面获取输入参数
    """

    # 清空屏幕
    reDraw.fig.clf()

    # 添加figure
    reDraw.plt = reDraw.fig.add_subplot(111)

    # 检查复选框是否选中
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regress_tree_utils.createTree(reDraw.rawDat, regress_tree_utils.modelLeaf, regress_tree_utils.modelErr, (tolS, tolN))
        yHat = regress_tree_utils.createForeCast(myTree, reDraw.testDat, regress_tree_utils.modelTreeEval)
    else:
        myTree = regress_tree_utils.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regress_tree_utils.createForeCast(myTree, reDraw.testDat)

    # 绘制拟合结果
    reDraw.plt.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    reDraw.plt.plot(reDraw.testDat, yHat, linewidth=2.0, c='red')
    reDraw.canvas.draw()


def main(root):
    # 标题
    root.title("回归树拟合演示")

    # 输入栏1, 叶子的数量
    tk.Label(root, text="叶子的数量").grid(row=1, column=0)
    global tolNentry
    tolNentry = tk.Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')

    # 输入栏2, 误差量
    tk.Label(root, text="误差量").grid(row=2, column=0)
    global tolSentry
    tolSentry = tk.Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0,'1.0')

    # 设置复选按钮
    global chkBtnVar
    chkBtnVar = tk.IntVar()
    chkBtn = tk.Checkbutton(root, text="Model Tree", variable = chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    # 设置提交的按钮
    tk.Button(root, text="确定", command=drawNewTree).grid(row=1, column=2, rowspan=3)

    # 退出按钮
    tk.Button(root, text="退出", fg="black", command=quit).grid(row=1, column=2)

    # 创建一个画板
    reDraw.fig = Figure(figsize=(5, 4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.fig, master=root)
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    # 加载数据并绘制图形
    reDraw.rawDat = np.mat(regress_tree_utils.load_data('../../../Data/RegressionTree/sine.txt'))
    reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    root = tk.Tk()
    main(root)
    root.mainloop()
