# -*- coding: utf-8 -*-
"""
create time : 2017-10-19 18:43:43
author : sk


"""

from __future__ import division  

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from util import *


# step1: read data 
data_X, data_y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, weights=[.5, .5], random_state=20) 
data_X = np.concatenate((data_X, np.ones(shape=[len(data_X),1])), axis=1)

# step2: define parameters for the model 
learning_rate = 0.01
n_epochs = 1000

# step3: define sigmoid function 
def sigmoid(x):
    return 1/(1+np.exp(-x))

# step4: create weights and bias 
w = np.random.random(size=data_X.shape[1])

# step5: gradient ascent
for i in range(n_epochs):
    w = w + learning_rate * (data_y-sigmoid(data_X.dot(w))).dot(data_X)

# step6: plot figure 
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid.dot(w[:-1])+w[-1])
probs = probs.reshape(xx.shape)
lr_figure(xx, yy, probs, data_X, data_y)




