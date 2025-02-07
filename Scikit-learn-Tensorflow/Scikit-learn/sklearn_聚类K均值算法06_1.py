#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/6 0006
@FILE: sklearn_聚类K均值算法06_1
@Author: Leo
"""

# 聚类K均值算法简介
"""
K均值聚类算法计算质心并进行迭代，直到找到最佳质心为止。
它假定群集的数目是已知的。
它也称为 扁平聚类算法。通过算法从数据中识别出的簇数用K均值中的" K"表示。

在此算法中，将数据点分配给群集，以使数据点和质心之间的平方距离之和最小。
聚类中较少的变化将导致相同聚类中更多的相似数据点。
"""

# 剧烈K均值算法步骤
"""
1.指定需要通过该算法生成的簇数K
2.随机选择K个数据点并将每个数据点分配给一个群集。
3.计算聚类质心
4.继续迭代一下步骤，直到找到最佳质心为止，这是将数据点分配给不在变化的簇
    4.1计算数据点和形心之间的平方距离之和
    4.2将每个书店分配给比其他群集（质心）更近的群集
    4.3通过获取该群集的所有数据点的平均值来计算群集的质心
K-均值遵循 期望最大化 方法来解决此问题。
期望步骤用于将数据点分配给最近的群集，而最大化步骤用于计算每个群集的质心。

使用K-means算法时，我们需要注意以下事项-
在使用包括K-Means的聚类算法时，建议对数据进行标准化，因为此类算法使用基于距离的度量来确定数据点之间的相似性。
由于K均值的迭代性质和质心的随机初始化，K均值可能会停留在局部最优值上，而可能不会收敛于全局最优值。因此，建议使用不同的质心初始化。
"""


# Pyhton实现
# 示例1--将首先生成包含4个不同Blob的2D数据集，然后将应用k-means算法查看结果
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.cluster import KMeans

# 生成2D，包含4个斑点
from sklearn.datasets.samples_generator import make_blobs
X,y_true = make_blobs(n_samples=400,centers=4,cluster_std=0.60,random_state=0)
plt.scatter(X[:,0],X[:,1],s=20)
plt.show()

# 制作一个Kmeans对象并提供聚类数量
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 绘制和可视化有k均值Python估计器选择的集群中心
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 20, cmap = 'summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()


# 示例2--将对简单数字数据集应用K均值聚类。 K-means将尝试在不使用原始标签信息的情况下识别相似的数字

# 从sklearn加载数字数据集并使其成为对象。我们还可以在此数据集中找到行数和列数，
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)        # 该数据集包含1797个具有64个特征的样本

# 执行聚类
kmeans = KMeans(n_clusters=10,random_state=0)
cluster = kmeans.fit_predict(digits.data)
# print(kmeans.cluster_centers_.shape)      # 具有64个特征的10个聚类

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

# 将学习的集群标签与在其中找到的真实标签匹配-
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]

# 检查准确性-
from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

"""
优缺点
优势
以下是K-Means聚类算法的一些优点-
这很容易理解和实施。
如果我们有大量变量，那么K均值将比层次聚类更快。
重新计算质心时，实例可以更改群集。
与分层聚类相比，更紧密的聚类由K均值形成。
缺点
以下是K-Means聚类算法的一些缺点-
预测簇的数量（即k的值）有点困难。
输出受初始输入（例如簇数（k值））的强烈影响
数据顺序将对最终输出产生重大影响。
它对重新缩放非常敏感。如果我们要通过归一化或标准化来重新缩放数据，那么输出将完全改变。
如果簇具有复杂的几何形状，则不利于进行聚类工作。"""


# K均值聚类算法的应用
# 聚类分析的主要目标是-
# 要从正在使用的数据中获得有意义的直觉。
# 先集群再预测将为不同的子组构建不同的模型。
# 为了实现上述目标，K均值聚类表现良好。可以在以下应用程序中使用-
# 市场细分
# 文档聚类
# 图像分割
# 图像压缩
# 客户细分
# 分析动态数据的趋势



