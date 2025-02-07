#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/6 0006
@FILE: sklearn_均值漂移算法06_2
@Author: Leo
"""


# 均值漂移算法简介
"""
这是用于无监督学习中的另一种强大的聚类算法。与K均值聚类不同，它没有做任何假设；因此它是一种非参数算法。
平均移位算法基本上是通过将数据点移向最高密度的数据点（即群集质心）来迭代的将数据点分配给群集
K-Means算法语Mean-Shift的区别在于，后一种算法无需实现指定聚类数，因为聚类数将由w.r.t数据算法确定
"""

# 均值漂移算法的工作
"""
1.从分配给他们自己的集群的数据点开始
2.该算法计算质心
3.新质心的位置将被更新
4.该过程被迭代并移至更高密度的区域
5.一旦质心到达无法进一步移动的位置，它将停止
"""

# Python实现
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets.samples_generator import make_blobs
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)  # 指定样本数，中心点，标准差
plt.scatter(X[:,0],X[:,1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']
for i in range(len(X)):
   plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
   marker = ".",color = 'k', s = 20, linewidths = 5, zorder = 10)
plt.show()

"""
优缺点
优势
以下是Mean-Shift聚类算法的一些优点-
它不需要像K-means或高斯混合中那样做出任何模型假设。
它还可以对非凸形状的复杂簇进行建模。
它只需要一个名为带宽的参数即可自动确定群集数。
没有像K-means中那样的局部最小值问题。
离群值没有问题。
缺点
以下是Mean-Shift聚类算法的一些缺点-
在高维情况下（簇数突然变化），均值漂移算法效果不佳。
我们无法直接控制集群的数量，但是在某些应用程序中，我们需要特定数量的集群。
它无法区分有意义的模式和无意义的模式。
"""
























