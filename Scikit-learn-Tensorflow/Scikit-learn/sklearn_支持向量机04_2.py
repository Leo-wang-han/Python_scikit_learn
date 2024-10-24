#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/10/24 0024
@FILE: sklearn_支持向量机04_2
@Author: Leo
"""
# 支持向量机
"""
支持向量机（SVM）是强大而灵活的监督式机器学习算法，可用于分类和回归。但通常，它们用于分类问题。
与其他机器学习算法相比，SVM具有其独特的实现方式。最近，由于它们能够处理多个连续和分类变量，它们非常受欢迎。
"""

# SVM工作
"""
SVM模型基本上是多维空间中超平面中不同类的表示。 
SVM将以迭代方式生成超平面，从而可以最大程度地减少误差。 
SVM的目标是将数据集分为几类，以找到最大边缘超平面（MMH）。
(个人理解，多条线能够分类，找到一条扩展延长最长的分类平面，平面边缘上的数据点是支持向量，存储在分类器的 support_vectors _属性中)
"""

# SVM内核
"""
在实践中，SVM算法是通过内核实现的，该内核将输入数据空间转换为所需的形式。
线性内核    K（x，xi）=sum（x∗xi）     kernel = 'linear'
多项式内核   k（X，Xi）=C+sum（X∗Xi） hatd 这里d是多项式的阶数，我们需要在学习算法中手动指定。
径向基函数（RBF）内核    K（x，xi）=exp（−gamma∗sum（x−xi hat2））    kernel = 'rbf', gamma =‘auto’（gamma：0-1）
Svc_classifier = svm.SVC（内核='线性'，C = C）.fit（X，y）  ，C是正则化参数，C=1
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# 使用原始数据绘制SVM边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]   # 将两个一维数组合并成一个二维数组

# 正则化参数值
C = 1.0

# 创建SVM分类器
Svc_classifier = svm.SVC(kernel="linear", C=C,).fit(X, y)
Z = Svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')











