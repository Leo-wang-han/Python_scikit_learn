#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/10/24 0024
@FILE: sklean_逻辑回归04_1
@Author: Leo
"""


# 逻辑回归的类型
# 逻辑回归是指具有二进制目标变量的二进制逻辑回归，但是可以通过它预测两类以上的目标变量
"""
二元或二项式
在这种分类中，因变量将只有两种可能的类型，即1和0。例如，这些变量可以表示成功或失败，是或否，赢或输等。

多项式
在这种分类中，因变量可以具有3种或更多可能的 无序 类型或无定量意义的类型。例如，这些变量可以表示"类型A"或"类型B"或"类型C"。

序号
在这种分类中，因变量可以具有3种或更多可能的 有序 类型或具有定量意义的类型。例如，这些变量可以表示"差"或"好"，"非常好"，"优秀"，并且每个类别的得分都可以为0、1、2、3。
"""

# 逻辑回归假设
"""
在深入研究逻辑回归的实现之前，我们必须了解以下关于相同的假设-
对于二进制逻辑回归，目标变量必须始终为二进制，并且期望结果由因子级别1表示。
模型中不应存在任何多重共线性，这意味着自变量必须彼此独立。
我们必须在模型中包括有意义的变量。
我们应该选择大样本量进行逻辑回归。
"""

# 回归模型
"""
二进制Logistic回归模型-Logistic回归的最简单形式是二进制或二项式Logistic回归，其中目标或因变量只能具有两种可能的类型1或0。
多项Logistic回归模型-Logistic回归的另一种有用形式是多项Logistic回归，其中目标或因变量可以具有3个或更多可能的 无序 类型，即没有定量意义的类型。
"""

# 多项逻辑回归
"""
逻辑回归的另一种有用形式是多项逻辑回归，其中目标或因变量可以具有3种或多种可能的 无序 类型，即没有定量意义的类型。
"""

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_iris()

# 定义特征矩阵（X) 和响应向量（y）
X = digits["data"]
y = digits["target"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

# 训练模型和预测
moudel = linear_model.LogisticRegression()
fit = moudel.fit(X_train, y_train)
predicted = moudel.predict(X_test)

# 精确度
score = metrics.accuracy_score(y_test, predicted)
print("accuracy:", score)

