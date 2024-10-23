#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/10/23 0023
@FILE: sklearn_分类算法04
@Author: Leo
"""
# 数据集
"""
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database 
sklearn.datasets
"""


# 分类简介
"""
分类可以定义为根据观测值或给定数据点预测类别或类别的过程.
在数学上，分类是从输入变量（X）到输出变量（Y）近似映射函数（f）的任务。
它基本上属于有监督的机器学习，其中还提供了目标以及输入数据集。
分类问题的一个例子是电子邮件中的垃圾邮件检测。只能有两类输出："垃圾邮件"和"无垃圾邮件"；因此，这是一个二进制类型分类。
"""

# 分类中学习者的类型
"""
懒惰的学习者：
训练时间少，预测时间多，等待测试数据；K近邻和基于案例的推理
渴望的学习者：
训练时间多，预测时间少，在存储训练数据后无需等待测试数据出现就构建分类模型；决策树，朴素贝叶斯和人工神经网络（ANN）
"""


# Python  构建分类器
import sklearn
from sklearn.datasets import load_breast_cancer   # 乳腺癌威斯康星州诊断数据库

# 加载数据集
data = load_breast_cancer()
# print(data.items())
label_names = data["target_names"]    # 恶性/良性---['malignant' 'benign'] ---标签
label = data["target"]                # 0/1
feature_names = data['feature_names'] # 特征
features = data['data']               # 特征数据
# print(label_names,label,feature_names,features,sep="\n")

# 数据划分为训练集和测试集
from sklearn.model_selection import train_test_split

# 特征数据划分问train，test; 标签数据划分为train_labels,test_labels
train, test, train_labels, test_labels = train_test_split(features,label,test_size=0.4,random_state=16)

# 构建模型
from sklearn.naive_bayes import GaussianNB   # 朴素贝叶斯算法

gnb = GaussianNB()                      # 构建模型
model = gnb.fit(train,train_labels)     # 模型数据训练  ---训练用于训练的特征数据和标签
predict = gnb.predict(test)             # 模型数据预测  ---预测用于预测的测试特征数据得到预测标签
print(predict)

# 寻找准确性
from sklearn.metrics import accuracy_score      # 准确性
accuracy = accuracy_score(test_labels,predict)  # 测试标签与预测标签的准确性
print(accuracy)

# 分类评估指标
"""
混乱矩阵:这是衡量分类问题性能的最简单方法，其中输出可以是两个或更多类型的课程

分类算法：
逻辑回归
支持向量机（svm)
决策树
朴素贝叶斯
随机森林
"""

# 应用程序
"""
语音识别
手写识别
生物特征识别
文档分类
"""


