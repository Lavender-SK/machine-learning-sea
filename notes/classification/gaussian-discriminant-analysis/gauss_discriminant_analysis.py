# -*- coding: utf-8 -*-
"""
create time : 2018-09-13 10:09:41
author : sk


"""

import numpy as np

class GaussDiscriminantAnalysis:
    def __init__(self):
        self.theta = None
        self.mu0 = None
        self.mu1 = None
        self.epsilon = None
    
    def fit(self, X, y):
        """ 极大似然估计-训练高斯判别分析模型-暂定二分类
        """
        # 估计 y
        self.theta = sum(y) / len(y)

        # 估计 第0类的均值  mu0
        self.mu0 = np.mean(X[y==0], axis=0)

        # 估计 第1类的均值 mu1
        self.mu1 = np.mean(X[y==1], axis=0)

        # 估计 协方差 epsilon
        Mu = np.array([self.mu0 if i == 0 else self.mu1 for i in y ])
        self.epsilon = np.dot((X-Mu).T, (X-Mu)) / len(y)
    
    def _predict(self, X):
        """ 计算 p(X|y)
        """
        epsilon_i = np.linalg.inv(self.epsilon)

        # 看第0类的结果
        y_0 = np.exp(-(1/2)*np.dot(np.dot((X-self.mu0), epsilon_i), (X-self.mu0).T))
        y_0 = [y_0[i,i] for i in range(len(y_0))]

        # 看第1类的结果
        y_1 = np.exp(-(1/2)*np.dot(np.dot((X-self.mu1), epsilon_i), (X-self.mu1).T))
        y_1 = [y_1[i,i] for i in range(len(y_1))]

        return y_0, y_1

    def predict_proba(self, X):
        """ 模型预测概率
        """
        y_0, y_1 = self._predict(X)
        result = [[i/(i+j), j/(i+j)] for i, j in zip(y_0, y_1)]
        return np.array(result)

    def predict(self, X):
        """ 模型预测
        """
        y_0, y_1 = self._predict(X)   
        # 综合比较
        return np.array([0 if i>j else 1 for i, j in zip(y_0, y_1)])
