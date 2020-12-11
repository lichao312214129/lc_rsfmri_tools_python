# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:14:53 2020

@author: lenovo
"""

# 导入相应的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 使用sklearn自带的鸢尾花数据
iris_dataset=load_iris()
x = iris_dataset["data"]
y = iris_dataset["target"]
x = x[y!=2,:]
y = y[y!=2]
n_sample = 100

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

#%% 开始机器学习
alpha = 0.01 # 学习率
max_cycles = 10000 # 最大循环次数
w_fit = [1,1,1,1]  # 初始化回归系数w

# 迭代100次，走100步看看能否到达山脚下
w_fit_all = []
loss_all = []
for index in range(max_cycles):
    z = np.dot(X_train, w_fit)  # 矩阵乘法
    yhat = 1.0/(1+np.exp(-z))   # 计算预测的y 
    loss = (-1/n_sample)*(np.sum((y_train*np.log(yhat)) + ((1-y_train)*(np.log(1-yhat)))))  # 损失函数
    delta_loss = (1/n_sample)*(np.dot(X_train.T, (yhat-y_train).T)) # 当前loss在w上的导数， 即对w求导
    print(f"{index}:loss={loss:.5f}, w_fit={w_fit}")
    w_fit = w_fit - alpha * delta_loss.T  # 更新w_fit

print(f"拟合出的权重为{w_fit}")

# 对测试集进行测试
z = np.dot(X_test, w_fit)  # 矩阵乘法
yhat = 1.0/(1+np.exp(-z))   # 计算预测的y 
yhat = np.int64(yhat>0.5)
acc = np.sum((yhat - y_test)==0)/len(yhat)
print(acc)


#%%  使用sklearn封装的逻辑回归
lr = LogisticRegression()
lr.fit(X_train, y_train)
yhat = lr.predict(X_test)
acc = np.sum((yhat - y_test)==0)/len(yhat)
print(acc)