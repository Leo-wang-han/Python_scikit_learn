#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2025/2/5 0005
@FILE: sklearn_线性回归05_1
@Author: Leo
"""

# 线性回归简介
"""
线性回归定义为统计模型，用于分析因变量与给定的一组自变量之间的线性关系。
变量之间的线性关系意味着当一个或多个自变量的值改变时，因变量的值也将相应改变
数学上：Y = m*X +b
Y是试图预测的因变量，X是我们用来进行预测的自变量，m是回归的斜率，表示X对Y的影响
"""

# 线性回归的类型
"""
简单线性回归SLR:Y = m*X +b
多元线性回归:h（xi）=b0+b1xi1+b2xi2 ::+ dotsm+bpxip
多个线性回归模型始终将数据中的误差称为残差误差，该残差会按以下方式更改计算方式:
h（xi）=b0+b1xi1+b2xi2+ dotsm+bpxip+ei
"""
# SLR实现
import numpy as np
import matplotlib.pyplot as plt

#定义一个函数计算SLR的重要值
def coef_estimation(x,y):   # 传入两个numpy数组
    n = np.size(x)
    m_x,m_y = np.mean(x),np.mean(y)     # x,y向量的平均值
    SS_xy = np.sum(y*x) - n*m_y*m_x     # x的交叉偏差
    SS_xx = np.sum(x*x) - n*m_x*m_x     # x的偏差
    # 计算回归系数
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return [b_0,b_1]

# 绘制回归线并预测响应向量
def plot_regression_line(x,y,b):
    plt.scatter(x,y,color="m",marker="o",s=30)
    y_pred = b[1]*x + b[0]
    plt.plot(x,y_pred,color="g")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# 多元线性回归实现
def m_linear_regression():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets,linear_model,metrics
    from sklearn.model_selection import train_test_split

    # 数据
    boston  = datasets.load_boston(return_X_y= False)       # 返回bunch对象
    x = boston.data
    y = boston.target
    #划分
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7,random_state=1)
    #训练
    reg = linear_model.LinearRegression()
    reg.fit(x_train,y_train)
    print('Coefficients: \n', reg.coef_)        # 斜率
    print('Variance score: {}'.format(reg.score(x_test, y_test)))       #R**2,决定系数
    plt.style.use('fivethirtyeight')
    plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color = "green", s = 10, label = 'Train data')
    plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color = "blue", s = 10, label = 'Test data')
    plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
    plt.legend(loc = 'upper right')
    plt.title("Residual errors")
    plt.show()


def main():
    # 简单线性回归SLR
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
    b = coef_estimation(x,y)
    print(f"Estimated coefficients:b_0 = {b[0]},b_1 = {b[0]}")
    plot_regression_line(x, y, b)
    m_linear_regression()

if __name__ == "__main__":
    main()
















# 假设
"""
以下是关于线性回归模型所建立的数据集的一些假设

多重共线性：线性回归模型假设中几乎没有多重共线性。基本上，当共变量或要素具有相关性时，就会发生多重共线性；
自相关：线性回归模型的另一个假设是数据中几乎没有自相关。基本上，当残差间存在依赖性时就会发生自相关；
变量之间的关系：线性回归模型假定响应变量和特征变量之间的关系必须是线性的

"""