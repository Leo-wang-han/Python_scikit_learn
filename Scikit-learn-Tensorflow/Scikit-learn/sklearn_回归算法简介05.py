#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/5 0005
@FILE: sklearn_回归算法简介05
@Author: Leo
"""

# 回归算法简介
"""
基于回归的任务的主要目标是针对给定的输入数据，预测输出标签或响应（连续的数值）。
基本上，回归模型使用输入数据热证（独立变量）及其对应的连续数值输出值（因变量或结果变量）来学习输入与对应输出之间的特定关联
"""

# 回归模型类型
"""
简单回归模型：这是最基本的回归模型，其中，预测是根据数据的一个单变量特征形成的；
多元回归模型：在该回归模型中，预测是根据数据的多个特征形成的
"""

# 在Python中构建回归器
# 下面构建基本的回归模型，该模型将使一条线适合数据，即线性该回归器
# 1.导包
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# 2.导入数据集
data = pd.read_csv("F:/study/Python/files_in/diabetes.csv",header=0)
print(data.columns.tolist())
# print(data.iloc[:100,9])
X,y = np.array(data.iloc[:,-3]).reshape(-1,1),np.array(data.iloc[:,-2]).reshape(-1,1)

# 3.将数据整理到训练和测试集中
training_sample = int(0.8 * len(X))
testing_samlpe = len(X) - training_sample
X_train,y_train = X[:training_sample],y[:training_sample]
X_test,y_test = X[training_sample:],y[training_sample:]
print(len(X_test),len(y_test))
print(X_test.shape,y_test.shape)
print(type(X_test),type(y_test))

# 4.模型评估与预测
reg_linear = linear_model.LinearRegression()
reg_linear.fit(X_train,y_train)
y_test_pred = reg_linear.predict(X_test)

# 5.绘图与可视化
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_test_pred,color="black",linewidth=2)
plt.xticks()
plt.yticks()
plt.show()

# 6.性能指标计算
print("Regressor model performance:")
print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error(MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 应用程序
"""
预测或预测分析：预测GDP,石油价格或简单地说随着时间的推移而变化的定量数据
优化:借助回归来优化业务流程，商定经理可以创建一个统计模型来了解顾客来访时间
错误纠正
经济学：预测供应、需求、消耗、库存投资等
财务：最小化风险组合

"""


