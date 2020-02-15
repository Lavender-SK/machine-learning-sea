# -*- coding: utf-8 -*-
"""
create time : 2018-05-30 13:42:54
author : sk


"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

class LDA(object):
    def __init__(self):
        self.__omega = None
        self.__sb = None
        self.__sw = None
        self.__mu = dict()
    
    def __split_class(self, X, y):
        ''' 拆分类
        '''
        split_json = dict()
        unique_y = np.sort(np.unique(y))
        for class_idx in unique_y:
            split_json[class_idx] = X[y==class_idx]

        return split_json

    def __cal_mu(self, array):
        ''' 计算每一个类的均值向量
        '''
        return np.mean(array, axis=0).reshape(-1,1)

    def __cal_within_scatter(self, X, y):
        ''' 计算类内离散度矩阵
        '''
        split_json = self.__split_class(X, y)

        for (k,v) in split_json.items():
            self.__mu[k] = self.__cal_mu(v)
        
        X_1 = list(split_json.values())[0]
        X_2 = list(split_json.values())[1]
        
        self.__sw = np.cov(X_1.T) + np.cov(X_2.T)

        return self.__sw

    def __cal_between_scatter(self, X, y):
        ''' 计算类间散度矩阵
        '''
        split_json = self.__split_class(X, y)

        for (k, v) in split_json.items():
            self.__mu[k] = self.__cal_mu(v)
        
        mu_1 = list(self.__mu.values())[0]
        mu_2 = list(self.__mu.values())[1]

        self.__sb = np.dot((mu_1-mu_2), (mu_1-mu_2).T)
        
        print(self.__mu)

        return self.__sb

    def fit(self, X, y):
        ''' 训练模型
        '''
        # 计算类间离散度矩阵
        self.__cal_between_scatter(X, y)
        # 计算类内离散度矩阵
        self.__cal_within_scatter(X,y)
        # 求两类的 mu
        mu_1 = list(self.__mu.values())[0]
        mu_2 = list(self.__mu.values())[1]
        # 求解参数向量
        self.__omega = np.dot(np.linalg.inv(self.__sw), (mu_1-mu_2))
        print(self.__omega)
    
    def predict(self, x):
        ''' 模型预测
        '''
        
        omega = self.__omega
        mu_1 = list(self.__mu.values())[0]
        mu_2 = list(self.__mu.values())[1]
        hat_mu_1 = np.dot(omega.T, mu_1)
        hat_mu_2 = np.dot(omega.T, mu_2)
        hat_x = np.dot(x, omega)
        
        if np.abs(hat_x[0][0]-hat_mu_1[0][0]):
            return 1
        else:
            return 2
    
    def draw_fig(self, X, y):
        ''' 画图
        ''' 
        
        split_json = self.__split_class(X,y)
        
        set_list = [('blue','s'),('red','o')]
        for item, s in zip(split_json.items(), set_list):
            plt.scatter(item[1][:,0], item[1][:,1],c=s[0],marker=s[1])
        
        plt.plot([0, 3.41528239*3], [0, 1.56810631*3])

        # plt.plot([1,2],[3,4])
        plt.show()


    #%%============================================================================
    # test
    #==============================================================================
    def test_cal_mu(self, array):
        return self.__cal_mu(array)

    def test_cal_between_scatter(self, X, y):
        return self.__cal_between_scatter(X,y)

    def test_cal_within_scatter(self, X, y):
        return self.__cal_within_scatter(X, y)

    def test_split_class(self, X, y):
        return self.__split_class(X, y)

    def test_cal_within_scatter(self, X, y):
        return self.__cal_within_scatter(X, y)

    def test_fit(self, X, y):
        return self.fit(X,y)

    def test_predict(self, x):
        return self.predict(x)
    
if __name__ == '__main__':
    lda = LDA()
    
    X = np.array(
        [[4,2],
         [2,4],
         [2,3],
         [3,6],
         [4,4],
         [9,10],
         [6,8],
         [9,5],
         [8,7],
         [10,8]]
    )

    y = np.array([1]*5+[2]*5)

    print('split class')
    print(lda.test_split_class(X, y))
    
    print('cal between scatter matrix')
    print(lda.test_cal_between_scatter(X,y))

    print('cal within scatter matrix')
    print(lda.test_cal_within_scatter(X,y))
    
    print('fit model')
    print(lda.test_fit(X,y))

    print('model predict')
    print(lda.test_predict(np.array([[4,3]])))

    print('sklearn lda')
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.predict(np.array([[4,3]])))

    print('draw figure')
    lda.draw_fig(X, y)
