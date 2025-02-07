#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/11/3 0003
@FILE: sklearn_决策树07
@Author: Leo
"""

# 决策树简介
"""
通常，决策树分析是一种可应用于许多领域的预测建模工具。
决策树可以通过一种算法方法构建，该算法可以根据不同条件以不同方式拆分数据集。
决策树是监督算法类别下最强大的算法。

分类决策树：在这种决策树中，决策变量是分类的。
回归决策树-在这种决策树中，决策变量是连续的。
"""

# 基尼系数
"""
首先，使用公式p**2 + q**2计算子节点的基尼系数，该公式是成功和失败概率的平方之和
接下来，使用该拆分的每个节点的加权Gini得分计算拆分的Gini指数
"""

# 拆分创建
"""
1.计算基尼得分
2.分割数据集
3.评估所有分割
"""

# 建树
"""
1.创建终端节点
最大树深度：树中根节点之后的最大节点数。一棵树达到最大深度时，即一棵树达到最大中断节点数时，我们必须停止天剑终端节点。
最小节点记录：可以定义为给定节点负责的最小训练模式数。
2.递归拆分
一旦创建了一个节点，我们就可以在每个数据组上递归地创建子节点，这些子节点是通过拆分数据集，一次有一次的调用同一函数而生成的
2.1预测
使用专门提供的数据行浏览决策树
2.2假设
"""

# Python实现Pima印度糖料病上实现决策树分类器
# 1.导包
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 2.导入数据集
# 怀孕、葡萄糖、BP、皮肤、胰岛素、Bmi、谱系、年龄、标签
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r"F:/study/Python/files_in/diabetes.csv")
pima.columns = col_names
# print(pima.head())

# 3.拆分数据集为特征和目标变量，训练集和测试集
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age',]
X = pima[feature_cols]      # 特征
Y = pima.label              # 目标变量

# 训练和测试
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=16)

# 4.训练模型、进行预测
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 混淆矩阵、分类报告、准确性得分
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
result = confusion_matrix(y_test,y_pred)
print(f"Confusion Matrix:{result}")
result1 = classification_report(y_test,y_pred)
print(f"Classifier Report:{result1}")
result2 = accuracy_score(y_test,y_pred)
print(f"Accuracy:{result2}")

