#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/1/20 0020
@FILE: sklearn_随机森林算法04_5
@Author: Leo
"""

# 简介
"""
随机森林是一种监督学些算法，可用于分类和回归。主要用于分类问题，这是一种集成学习，
可以通过对结果进行平均来减少过拟合。
1.首先，新给定的数据集中选择随机样本
2.接下来，该算法将为每个样本构造一个决策树。然后从每个决策树种获取预测结果
3.在此步骤中，将对每个预测结果进行投票
4.最后，选择投票最多的预测结果作为最终预测结果
"""

# Python实现
# 导包
import numpy as np
