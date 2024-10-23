#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/6/13 0013
@FILE: 数据处理02
@Author: Leo
"""
import pandas as pd
import numpy as np
path = r"F:/study/Python/files_in/0518-0522.csv"
data = pd.read_csv(path)
data = data[["fyll", "glmqlyl","hfqrfwd","lfll","lfwd","lfyl","mqrz","rfwd1"]]
# 1.缩放 --- MinMaxScaler类
from sklearn import preprocessing
array = data.values
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))  # 范围0-1
data_rescaled = data_scaler.fit_transform(array)
np.set_printoptions(precision=1)
print ("\nScaled data:\n", data_rescaled[0:10])

# 2.规范化Normalizer ---稀疏矩阵
# 将每行数据重新缩放为长度为1
"""
L1标准化
L2标准化
"""

# 3.二值化 -- Binarizer类（将数据转化成二进制值）
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.5).fit(array)   # 阀值0.5
Data_binarized = binarizer.transform(array)
print ("\nBinary data:\n", Data_binarized [0:5])


# 4.标准化 --- StanderdScaler(平均值=0，标准偏差（SD)=1)
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)

# 5.标签编码
# 输入标签
input_labels = ['red','black','red','green','black','yellow','white']
# 创建标签编码器并对其训练，生成编码
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
# 对随机标签/编码有序列表进行编码解码检查性能
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels) # 编码
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values) # 解码
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))