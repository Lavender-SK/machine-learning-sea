# -*- coding: utf-8 -*-
"""
create time : 2017-10-20 14:31:36
author : sk


"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def create_data(function=lambda x:x+10*np.sin(0.3*x), size=200, random_scale=1.5):
    xx = np.arange(1, size+1)
    yy = function(xx) + stats.norm.rvs(size=size, loc=0, scale=random_scale)
    return xx,yy
    
    
def figure_data(xx, yy, is_save=False, save_name='test.png'):
    xlim = (np.min(xx), np.max(xx))
    ylim = (np.min(yy), np.max(yy))
    
    # plot figure 
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(xx, yy, color='red')
    ax.set(#aspect='equal', 
           xlim=xlim, ylim=ylim, 
           xlabel='$x$', ylabel='$y$',
           title='$y=f(x)$')
    ax.grid()
    
    if is_save:
        fig.savefig(save_name)
    plt.show()

def figure_model(data_X, data_y, data_pre_X, data_pre_y, is_save=False, 
                 save_name='test.png', model_info=''):
    xlim = (np.min(data_X), np.max(data_X))
    ylim = (np.min(data_y), np.max(data_y))
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(data_X, data_y, color='blue')
    ax.plot(data_pre_X, data_pre_y, color='red')
    ax.set(xlim=xlim, ylim=ylim, xlabel='$x$', ylabel='$y$', title='$y=f(x)$ '+model_info)
    ax.grid()
    
    if is_save:
        fig.savefig(save_name)
    
    plt.show()


if __name__ == '__main__':
    data_X, data_y = create_data()
    figure_data(data_X, data_y)






