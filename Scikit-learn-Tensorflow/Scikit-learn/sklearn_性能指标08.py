#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/7 0007
@FILE: sklearn_性能指标08
@Author: Leo
"""

# 分类问题绩效指标
"""
混淆矩阵（Confusion Matrix)
Confusion Matrix
[[16  0  0]
 [ 0 22  1]
 [ 0  3 18]]

1.混淆矩阵结构
混淆矩阵的行表示实际标签，列表示预测标签。对于你的 3 分类问题，矩阵的结构如下：

            预测类别1	预测类别2	预测类别3
实际类别1	    16	        0	        0
实际类别2	    0	        22	        1
实际类别3	    0	        3	        18


2.逐行解读
第一行 [16, 0, 0]：
实际类别为 1 的样本共有 16 个。
模型将这 16 个样本全部正确预测为类别 1。
类别 1 的预测完全正确。

第二行 [0, 22, 1]：
实际类别为 2 的样本共有 23 个（22 + 1）。
模型将其中 22 个正确预测为类别 2。
有 1 个样本被错误预测为类别 3。
类别 2 的预测准确率较高，但有少量错误。

第三行 [0, 3, 18]：
实际类别为 3 的样本共有 21 个（3 + 18）。
模型将其中 18 个正确预测为类别 3。
有 3 个样本被错误预测为类别 2。
类别 3 的预测准确率较高，但有少量错误。


3.关键指标计算
3.1 准确率(Accuracy)
Accuracy = (正确预测的总数/总样本数)=（16+22+18）/(16+22+18+1+3)=93.33%

3.2 每个类别的精确率(Precision)
表示模型预测为某一类别的样本中，实际属于该类别的比例：
    类别1：Precision1 = 16/(16+0+0)=100%
    类别2：Precision2 = 22/(0+22+1)=95.65%
    类别3：Precision3 = 18/(0+3+18)=85.71%

3.3 每个类别的召回率（Recall）
召回率表示实际属于某一类别的样本中，被模型正确预测的比例：
    类别1：Recall1 = 16/(16+0+0)=100%
    类别2：Recall2 = 22/(0+22+3)=88%
    类别3：Recall3 = 18/(0+1+18)=94.74%
    
    
3.4 F1分数（F1 Score)
F1 分数是精确率和召回率的调和平均值：
    F1 = 2 * (Precision * Recall)/(Precision + Recall)
    
3.5 AUC(ROC曲线下的面积)
通过在各种阈值下绘制TPR（真阳性率）即灵敏度或召回率与FPR（假阳性率）

3.6 LOGLOSS(对数损失)
称为Logistic回归损失或交叉熵损失。它基本上是根据概率估计定义的，并衡量分类模型的性能，其中输入是介于0和1之间的概率值。
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report # 模型评估报告
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

X_actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]       # 实际值
Y_predic = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0]       # 预测值

print("Confusion Matrix:",confusion_matrix(X_actual,Y_predic))
print("Accuracy Score:",accuracy_score(X_actual,Y_predic))
print("Classifier Report:",classification_report(X_actual,Y_predic))
print("AUC-ROC:",roc_auc_score(X_actual,Y_predic))
print("lOGLOSS:",log_loss(X_actual,Y_predic))


# 回归问题的绩效指标
"""
1.均方误差（Mean Squared Error,MSE)
预测值与实际值之差的平方和

2. 均方根误差（Root Mean Squared Error, RMSE）
MSE 的平方根，将误差恢复到原始单位

3.平均绝对误差（MAE）
预测值与实际值之差的绝对值的平均值。

4. 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）
预测值与实际值之差的绝对值占实际值的百分比的平均值。

5. 决定系数（R², R-squared）
表示模型对目标变量变异性的解释程度。

6. 调整决定系数（Adjusted R²）
对 R² 进行调整，考虑模型中特征的数量。

7. 解释方差（Explained Variance Score）
衡量模型对目标变量变异性的解释能力

MSE 和 RMSE：适用于对较大误差敏感的场景。
MAE：适用于对异常值不敏感的场景。
MAPE：适用于需要百分比误差的场景。
R² 和 Adjusted R²：适用于评估模型的整体解释能力。
Explained Variance：适用于评估模型对变异性的解释能力。
"""
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
X_actual = [5, -1, 2, 10]
Y_predic = [3.5, -0.9, 2, 9.9]

print ('MSE =',mean_squared_error(X_actual, Y_predic))
print ('MAE =',mean_absolute_error(X_actual, Y_predic))
print ('R Squared =',r2_score(X_actual, Y_predic))