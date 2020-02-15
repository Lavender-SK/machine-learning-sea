# -*- coding: utf-8 -*-
"""
create time : 2017-10-20 16:00:41
author : sk


"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from util import *

def locally_weighted_lr(tau, data_X, data_y, pre_x):
    """ Locally weighted linear regression
        args:
            tau:
                the bandwidth parameter, float 
            data_X:
                train data_X, shape=(n_samples, n_features), n_samples is the number of samples, n_features is the 
                number of features
            data_y:
                train data_y, shape=(n_samples,), n_samples is the number of samples
            pre_x:
                predicted input data, shape=(1,n_features), n_features is the number of features
        return:
            pre_y:
                the out result, float
    """
    
    # get sample datas
    if data_X.ndim == 1: 
        data_X = data_X.reshape(len(data_X),-1)
    n_samples, n_features = data_X.shape
    
    data_X = np.concatenate((data_X, np.ones(shape=[n_samples,1])), axis=1)
    pre_x = np.concatenate((pre_x.reshape(1,n_features), np.ones([1,1])), axis=1)
    
    # get non-negative valued weights 
    weights = np.exp(-np.sum((data_X-pre_x)*(data_X-pre_x),axis=1) / (2*tau*tau))
    
    # get the theta
    W = np.diag(weights)
    theta = np.linalg.inv(data_X.T.dot(W).dot(data_X)).dot(data_X.T).dot(W).dot(data_y) 
    
    # get the pre_y 
    return pre_x.dot(theta)


if __name__ == '__main__':
    data_X, data_y = create_data()

    for tau in np.arange(1,11,1)[::-1]:
        data_y_pre = np.array([locally_weighted_lr(tau, data_X, data_y, pre_x) for pre_x in data_X])
        figure_model(data_X, data_y, data_X, data_y_pre, model_info='tau='+str(tau))


