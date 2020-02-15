# -*- coding: utf-8 -*-
"""
create time : 2017-10-19 17:49:46
author : sk


"""

#%%============================================================================
# import module
#==============================================================================
import util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#%%============================================================================
# tensorflow logistic regression
#==============================================================================
# step1 : read data
data_X, data_y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, weights=[.5, .5], random_state=20) 

# step1 : Define parameters for the model
learning_rate = 0.01
batch_size = 200
n_epochs = 1000

# step2 : create placeholders for features and labels
X = tf.placeholder(tf.float32, [None,2], name='X')
y = tf.placeholder(tf.float32, [None,1], name='Y')

# step3 : create weights and bias
W = tf.Variable(tf.random_normal(shape=[2,1], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1,1]), name='bias')

# step4 : predict y from X and w, b
logits = tf.matmul(X, W) + b 
probs = tf.nn.sigmoid(logits)

# step5: define loss function
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy) 

# step6: define training op
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# step7: train and test the model 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # train the model
    for _ in range(n_epochs):
        X_batch, Y_batch = data_X, data_y.reshape(200,1)
        sess.run([optimizer,loss], feed_dict={X: X_batch, y:Y_batch})
    
    # plot figure 
    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, loss_batch, probs_batch = sess.run([optimizer, loss, probs], feed_dict={X: grid, y: np.zeros([len(grid),1])})
    probs = probs_batch.reshape(xx.shape)
    util.lr_figure(xx, yy, probs, data_X, data_y)

