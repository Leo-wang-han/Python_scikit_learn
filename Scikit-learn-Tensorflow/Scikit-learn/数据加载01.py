#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/6/13 0013
@FILE: 数据加载01
@Author: Leo
"""

"""
数据加载
"""
# 加载CSV文件
# 1.标准库加载
import csv
import numpy as np

path = "F:/study/Python/files_in/0518-0522.csv"
with open(path, "r",encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    headers = next(reader)
    data = list(reader)
    data = np.array(data)
# print(headers)


# 2.使用Numpy加载CSV
# 3.使用Pandas加载CSV
from pandas import read_csv
path = r"F:/study/Python/files_in/0518-0522.csv"
data = read_csv(path)


"""
数据统计
"""
# 查看数据维度
print(data.shape)

# 获取数据属性
print(data.dtypes)

# 数据描述
"""
总数、平均值、标准偏差、
最低价值、最大值、25%、中位数、75%
"""
print(data.describe())

# 获取某个类别的观察值数量--类分布
count = data.groupby("crvzc").size()
print(count)

# 查看属性之间的关联(高度相关性，对某些机器学习算法的性能将很差)
"""
系数值=1 变量之间完全正相关
系数值=-1 变量之间完全负相关
系数值=0 变量之间完全没有相关性
"""
import pandas as pd
pd.set_option("display.width", 100)
pd.set_option("precision", 2)
correlations = data.corr(method= "pearson")
print(correlations)

# 审查属性分布的偏斜 ---接近0，偏斜小
print(data.skew())


"""
可视化数据
"""
# 单变量 --- 每个属性分布
"""
直方图
密度图
箱型图（box)、晶须图（whisker)
"""
#多个变量相互作用
"""
相关矩阵图  --- 相关性
散点图 --- 一个变量受另一个变量的影响程度
"""








