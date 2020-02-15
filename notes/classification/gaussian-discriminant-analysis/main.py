# -*- coding: utf-8 -*-
"""
create time : 2018-09-13 14:07:47
author : sk


"""

from create_data import gen_data, plot_data, plot_decision_boundary
from gauss_discriminant_analysis import GaussDiscriminantAnalysis

# 创建训练数据集 X, y
X_train, y_train = gen_data()
plot_data(X_train, y_train)

# 训练模型
gauss_model = GaussDiscriminantAnalysis()
gauss_model.fit(X_train, y_train)

print(gauss_model.theta)
print(gauss_model.mu0)
print(gauss_model.mu1)
print(gauss_model.epsilon)

# 模型在训练数据集上的效果
y_hat = gauss_model.predict(X_train)
print(sum(y_train==y_hat) / len(y_hat))

y_proba_hat = gauss_model.predict_proba(X_train)

# 绘制模型分界面
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
plot_decision_boundary(x_min, x_max, gauss_model)
