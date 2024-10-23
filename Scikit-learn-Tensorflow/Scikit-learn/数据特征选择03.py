#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/10/22 0022
@FILE: 数据特征选择03
@Author: Leo
"""
# 1.数据特征选择的重要性
"""
在数据建模之前，
减少过拟合，提高ML模型的准确性，减少训练时间
"""

# 2.功能选择技巧
# 2.1单变量选择 (统计特征与预测变量具有最强的关系)
# --- SelectKBest()类（基于统计检验方法来给每个特征打分，选得分最高的k个特征）
from sklearn.feature_selection import SelectKBest
from  sklearn.datasets import load_iris   # 鸢尾花数据集
from sklearn.feature_selection import chi2    # 卡方检验
import numpy as np

iris = load_iris()
# 划分数据和特征
X, y = iris.data, iris.target
# 利用卡方检验，通过SelectKBest()方法选择最好的4个特征
selector = SelectKBest(score_func=chi2, k=4)
# 对划分的数据进行训练
fit  = selector.fit(X, y)

np.set_printoptions(precision=2)
print(fit.scores_)
# 选择功能
featured_data = fit.transform(X)
print ("\nFeatured data:\n", featured_data[0:4])


# 3.消除递归特征 （以递归方式删除属性，并使用剩余的属性构建模型）
# ---递归特征消除（RFE)   RFE类
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
path = "F:/study/Python/files_in/diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(path, names = names,encoding="utf8")
# array = data.values

# # 划分数据
# X = array[1:, 0:8]
# y = array[:, 8]
# # print(y)
# # # 从数据集中选择最佳功能
# model = LogisticRegression()
# rfe = RFE(model, 3)  # 选择前3个功能标记为1
# fit = rfe.fit(X, y)
# print("Number of Features: %d")
# print("Selected Features: %s")
# print("Feature Ranking: %s")


# 4.主成分分析（PCA) --- PCA类
"""
PCA，通常称为数据约简技术，是一种非常有用的特征选择技术，因为它使用线性代数将数据集转换为压缩形式
"""
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 5)    # 选择最佳3个主要成分
# fit = pca.fit(X)
# print("Explained Variance:")
# print(fit.components_)


# 5.功能重要性 --- ExtraTreeClassifier类
"""
特征重要性技术用于选择重要性特征。它基本上使用训练有素的监督分类器来选择要素。
从输出中，我们可以看到每个属性都有分数。得分越高，该属性的重要性就越高。
"""
# from sklearn.ensemble import ExtraTreesClassifier
# path = r'F:/study/Python/files_in/diabetes.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pd.read_csv(data, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]   # 所有行下标为8的元素
#
# model = ExtraTreesClassifier()
# model.fit(X, Y)
# print(model.feature_importances_)



import traceback
traceback.print_exc()



















