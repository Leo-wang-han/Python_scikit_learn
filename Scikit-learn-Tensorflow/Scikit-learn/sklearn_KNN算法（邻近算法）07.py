#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/7 0007
@FILE: sklearn_KNN算法（邻近算法）07
@Author: Leo
"""

# KNN算法简介
"""
K近邻算法是一种监督的Ml算法，可用于分类以及回归预测问题。
但是，他主要用于行业中的分类预测问题
惰性学习算法-KNN：是一种惰性学习算法，因为他没有专门的训练阶段，并且在分类时将所有数据用于训练
非参数学习算法-KNN:是一种非参数学习算法，因为它不假定基础数据
"""

# KNN算法的工作
"""
K近邻算法使用“特征相似度”来预测新数据点的值，这进一步意味着，将根据新数据点与训练集中
各点的匹配程度为该新数据点分配一个值。
1.加载训练以及测试数据
2.选择K值，即最近的数据点。K可以是任何证书
3.对于测试数据中的每个点，执行一下操作
    3.1借助以下任意一种方法来计算测试数据与每一行训练数据之间的距离：欧几里得距离，曼哈顿距离或韩明距离
    3.2基于距离值，将它们按升序排序
    3.3接下来，它将从排序后的数组中选择前K行
    3.4现在，它将基于这些行中最常见的类别为测试点分配一个类别
4.结束
"""

# Python实现
# KNN作为分类器

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# web下载数据集
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# 为数据集分配列名
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# 读取数据
dataset = pd.read_csv(path, names = headernames)
# print(dataset.head())

# 划分数据
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)

# 数据缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)     # fit方法对训练集X_train进行拟合，计算出每个特征的均值和标准差
X_train = scaler.transform(X_train)     # transform方法对训练集X_train进行转换，将每个特征值减去均值，然后除以标准差，得到标准后的训练集
X_test = scaler.transform(X_test)

# KNeighborsClassifier训练模型
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)      # 设置邻居数为8
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# 执行打印结果
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)



# KNN作为回归器

import numpy as np
import pandas as pd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv(path, names = headernames)
array = data.values
X = array[:,:2]
Y = array[:,2]

# KNeighborsRegressor拟合模型
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors=20)
knnr.fit(X,Y)
y_knnr = np.power(knnr.predict(X),2).mean()

print ("The MSE is:",y_knnr)


"""
KNN的优缺点
专业人士
这是一种非常简单的算法，可以理解和解释。
这对于非线性数据非常有用，因为该算法中没有关于数据的假设。
这是一种通用算法，我们可以将其用于分类和回归。
它具有相对较高的准确性，但是有比KNN更好的监督学习模型。
缺点
这是一种计算上有点昂贵的算法，因为它存储了所有训练数据。
与其他监督学习算法相比，需要高存储容量。
大N时预测速度很慢。
它对数据规模以及不相关的功能非常敏感。
KNN的应用
以下是可以成功应用KNN的一些领域-
银行系统
KNN可以在银行系统中用于预测个人适合贷款审批的天气吗？该个人是否具有与违约者相似的特征？
计算信用等级
通过与具有相似特征的人进行比较，可以使用KNN算法来查找个人的信用等级。
投票
借助KNN算法，我们可以将潜在选民分为多个类别，例如"将投票"，"将不投票"，可以使用KNN算法的其他领域是语音识别，手写检测，图像识别和视频识别。
"""


















