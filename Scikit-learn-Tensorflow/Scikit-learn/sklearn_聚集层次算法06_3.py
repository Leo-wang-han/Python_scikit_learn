#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/6 0006
@FILE: sklearn_聚集层次算法06_3
@Author: Leo
"""

# 聚集层次算法简介
"""
分层聚类是另一种无监督的学习算法，用于将具有相似特征的未标记数据点分组在一起。
分层聚类算法分为以下两类:
1.聚集层次算法：在剧集层次算法中，每个数据点都被视为单个群集，然后依次合并或聚集（之下而上）群集对。
群集的层次结构表示为树状图或树状结构
2.分割分层算法：在分割分层算法中，所有数据点都被视为一个大聚类，并且聚类的过程涉及将（大自上而下的方法）划分为一个大聚类聚集成各种小集群
"""

# 执行聚集层次聚类的步骤
"""
1.将每个数据点视为单个群集。因此，开始时我们将拥有K个群集。开始时，数据点的数量也将为K。
2.在这一步中，我们需要通过连接两个壁橱数据点来形成一个大集群。这将总共产生K-1个簇。
3.要形成更多集群，我们需要加入两个壁橱集群。这将导致总共有K-2个集群。
4.要形成一个大集群，请重复上述三个步骤，直到K变为0，即不再有要连接的数据点。
5.制作了一个大簇之后，将根据问题使用树状图将其分为多个簇。
"""

# 树状图在聚集层次聚类中的作用
# 如我们在最后一步中讨论的那样，一旦大集群形成，树状图就开始发挥作用。根据我们的问题，将使用树状图将群集分为相关数据点的多个群集
import matplotlib.pyplot as plt
import numpy as np


# 绘制数数据点
X = np.array([[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[87,96],])
labels = range(1, 11)
plt.figure(figsize = (10, 7))
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X[:,0],X[:,1], label = 'True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
   plt.annotate(label,xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')
plt.show()


# 使用Scipy库来绘制数据点的树状图
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize = (10, 7))
dendrogram(linked, orientation = 'top',labels = labelList,
   distance_sort ='descending',show_leaf_counts = True)
plt.show()

# 现在，一旦形成大簇，就选择了最长的垂直距离。然后通过一条垂直线绘制一条线，如下图所示。当水平线与蓝线在两个点处相交时，簇的数量将为两个。

# 接下来，我们需要导入用于聚类的类，并调用其fit_predict方法来预测聚类。
# 我们正在导入 sklearn.cluster 库的 AgglomerativeClustering 类-
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(X)
plt.scatter(X[:,0],X[:,1], c = cluster.labels_, cmap = 'rainbow')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import read_csv
path = r"F:/study/Python/files_in/diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
data.shape
(768, 9)
print(data.head())

patient_data = data.iloc[:, 3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (10, 7))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(data, method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')  # 聚类数量为4，欧式距离，连接方式
cluster.fit_predict(patient_data)
plt.figure(figsize = (10, 7))
plt.scatter(patient_data[:,0], patient_data[:,1], c = cluster.labels_, cmap = 'rainbow')




