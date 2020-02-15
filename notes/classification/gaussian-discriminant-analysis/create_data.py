# -*- coding: utf-8 -*-
"""
create time : 2018-09-13 10:38:23
author : sk


"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles


def gen_data():
    """ 生成数据
    """
    X1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=1, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=200, n_features=2, n_classes=1, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2+1))

    return X, y


def plot_data(X, y):
    """ 画出生成的数据
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"
    
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, s=20, edgecolor='k', label="Class %s" % n)
    
    plt.show()


def plot_decision_boundary(x_min, x_max, model):
    """ 绘制高斯判别分析分界面
    """
    xx, yy = np.mgrid[x_min:x_max:.2, x_min:x_max:.2]
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(grid)
    print(pred)

    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"
    
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(pred==i)
        plt.scatter(grid[idx, 0], grid[idx, 1], c=c, edgecolor='k')
    
    plt.show()


if __name__ == "__main__":
    X, y = gen_data()
    plot_data(X, y)
