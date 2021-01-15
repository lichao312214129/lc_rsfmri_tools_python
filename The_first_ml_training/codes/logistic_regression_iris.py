#!/usr/bin/env python
# coding: utf-8

# In[102]:
# 导入相应的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC

from training1 import training

# In[104]:
# 使用sklearn自带的鸢尾花数据
iris_dataset = load_iris()
x = iris_dataset["data"]
y = iris_dataset["target"]
x = x[y!=2,:]
y = y[y!=2]

# In[95]:

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(np.shape(y_train))
print(np.shape(X_train))

# In[106]:
w_fit = training(X_train, y_train)


# In[79]:

print(w_fit)


# In[80]:


# 对测试集进行测试
z = np.dot(X_test, w_fit)  # 矩阵乘法
yhat = 1.0/(1+np.exp(-z))   # 计算预测的y 
yhat = np.int64(yhat>0.5)  # 预测概率大于0.5视为正类（versicolor），小于0.5为负类（setosa）
acc = np.sum((yhat - y_test)==0)/len(yhat)  # 计算准确度，即分类正确比例
# print((yhat - y_test))
# print((yhat - y_test)==0)
print(acc)

#%%  使用sklearn封装的逻辑回归
lr = LinearSVC()
lr.fit(X_train, y_train)
yhat = lr.predict(X_test)
acc = np.sum((yhat - y_test)==0)/len(yhat)
print(yhat > 0.8)
print(np.int64(yhat > 0.8))
print(y_test)


# In[68]:


get_ipython().run_line_magic('pinfo', 'LinearSVC')


# In[39]:


a = [1,2,3]


# In[42]:


a[0:2]


# In[45]:


np.exp(-100)


# In[ ]:




