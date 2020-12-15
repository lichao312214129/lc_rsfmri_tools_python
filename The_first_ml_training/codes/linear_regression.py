# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:14:53 2020

@author: lenovo
"""

# 导入相应的包
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据集
np.random.seed(666)
x = np.random.randn(10,1)  #随机生成10个数, 作为自变量
w_true = 0.5  # 自定义系数W=0.5
y_true = x*w_true  # y = wx

# 查看x和y的关系
plt.scatter(x, y_true)
plt.xlabel("x")
plt.ylabel("y")

#%% 开始机器学习
alpha = 0.1 # 学习率
max_cycles = 200 # 最大循环次数
w_fit = 1  # 初始化回归系数w

# 迭代100次，走100步看看能否到达山脚下
w_fit_all = []
loss_all = []
for index in range(max_cycles):
    yhat = w_fit * x   # 计算预测的y
    loss = np.power((yhat-y_true), 2).mean()/2  # 计算当前的损失
    delta_loss = (x*(yhat-y_true)).mean() # 当前loss在w上的导数， 即对w求导
    print(f"{index}:loss={loss:.5f}, w_fit={w_fit}")
    w_fit = w_fit - alpha * delta_loss  # 更新w_fit

print(f"拟合出的权重为{w_fit}")
