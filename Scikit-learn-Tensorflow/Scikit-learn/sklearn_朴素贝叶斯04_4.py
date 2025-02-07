#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/1/20 0020
@FILE: sklearn_朴素贝叶斯04_4
@Author: Leo
"""

# 朴素贝叶斯简介
"""
朴素贝叶斯算法是一种基于贝叶斯定理的分类技术，其中强假设所有预测变量彼此独立。
简而言之，假设是某个类中某个要素的存在独立于同一类中其他任何要素的存在。
P（L | features）\:= \:\ frac {P（L）P（features | L）} {P（features}}
在这里，（𝐿|𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠）是类别的后验概率。类别的后验概率是指给定某个特征向量后，某个类别出现的概率:P(A|B) = P(B|A) * P(A) / P(B)
𝑃（𝐿）是类别的先验概率。
𝑃（𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠|𝐿）是似然度，它是给定类别的预测变量的概率。
𝑃（𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠）是预测变量的先验概率。指在观察到任何数据之前，预测变量每个可能取值的概率
"""

# 在Python中使用朴素贝叶斯构建模型
"""
1.高斯朴素贝叶斯：最简单的朴素贝叶斯分类器，其假设是每个标签的数据均来自简单的高斯分布
2.多项式朴素贝叶斯：其中的特征被认为是从简单的多项式分布中得出的。这种朴素的贝叶斯最适合代表离散计数的功能
3.伯努利朴素贝叶斯：其中的特征被假定为二进制。
"""


# 高斯朴素贝叶斯模型
#导库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# 生成高斯分布的数据
from sklearn.datasets import make_blobs
# 300样本，每个样本2个特征，2个簇，随机种子，标准差
X, y = make_blobs(300,2,centers=2,random_state=2,cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='viridis')
# plt.show()

# GaussionNB模型
from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB = model_GNB.fit(X,y)

# 生成新数据之后进行预测
rng = np.random.RandomState(0)      # 创建一个随机数生成器，随机数种子设定为0
Xnew = [-6,-14] + [14,18]*rng.rand(2000,2)  # 起始值与随机数相加
# print(Xnew[0])
ynew = model_GNB.predict(Xnew)

# 绘制新数据以查找其边界
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="summer")
lim = plt.axis()        # 获取当前最标轴范围(-5.864572514057202, 5.478233350869923, -14.00968321776786, 4.460699874510104)
# print(lim)
plt.scatter(Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = 'summer', alpha = 0.1)     # alpha设置每个点的透明度
plt.axis(lim)

# 找到第一和第二个标签的后验概率
yprob = model_GNB.predict_proba(Xnew)
yprob[-10:].round(3)
print(yprob)