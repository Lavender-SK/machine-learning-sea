# -*- coding: utf-8 -*-
"""
create time : 2017-10-19 18:45:24
author : sk


"""

from __future__ import division  

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification

#%%============================================================================
# plotting decision boundary of logistic regression
# https://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression
#==============================================================================
def lr_figure(xx, yy, probs, X, y, is_save=False, save_name='test.png'):
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    
    if is_save:
        f.savefig(save_name)
    
    plt.show()

def lr_decision_boundary(xx, yy, probs, X, y, is_save=False, save_name='test.png'):  
    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    
    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    if is_save:
        f.savefig(save_name)
    
    plt.show()

def sigmoid_figure(xlim=(-5,5), is_save=False, save_name='test.png'):
    # data for plotting 
    xx = np.arange(xlim[0],xlim[1],(xlim[1]-xlim[0])/2000)
    yy = sigmoid(xx)
    
    # plot figure 
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(xx, yy)
    ax.set(#aspect='equal', 
           xlim=xlim, ylim=(0,1), 
           xlabel='$x$', ylabel='$sigmoid(x)$',
           title='$f(x)=sigmoid(x)$')
    ax.grid()

    if is_save:
        fig.savefig(save_name)

    plt.show()

#%%============================================================================
# sigmoid function
#==============================================================================
def sigmoid(x):
    return 1/(1+np.exp(-x))

#%%============================================================================
# main function
#==============================================================================
if __name__ == "__main__":
    sigmoid_figure()


