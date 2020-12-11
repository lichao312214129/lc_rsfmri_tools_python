# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:55:30 2020
使用支持向量机分类算法，根据身高和体重来判断是否为肥胖
肥胖的金标准为bmi > 28
@author: Li Chao
"""

# 导入相应的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.svm import   SVC

# 生成训练数据集
np.random.seed(666)
height_train = np.random.random(200) + 1 # 身高
weight_train = np.random.random(200)*100  # 体重
x_train = np.vstack([height_train, weight_train]).T
bmi_train = np.array(weight_train)/np.power(height_train,2)
y_train = np.int64(bmi_train > 28)  # bmi大于28为肥胖

# 生成测试集
np.random.seed(888)
height_test = np.random.random(100) + 1 # 身高
weight_test = np.random.random(100)*100  # 体重
x_test = np.vstack([height_test, weight_test]).T
bmi_test = np.array(weight_test)/np.power(height_test,2)
y_test = np.int64(bmi_test > 28)  # bmi大于28为肥胖

# 查看训练集数据
x_train_ = pd.concat([pd.DataFrame(y_train), pd.DataFrame(x_train)], axis=1)
x_train_.columns = ["肥胖","身高","体重"]

# 训练模型
svm = SVC(kernel="linear", max_iter=1000)
svm.fit(x_train, y_train)



# 测试
w = svm.coef_  # 获取模型的权重
b = svm.intercept_  # 获取模型的截距
pred_value = np.dot(x_test, w.T) + b  # 各个样本点的预测分数 y = w*x +b

y_pred = np.int64(pred_value > 0)  # 预测值大于0，即说明其为正类
y_pred = y_pred.reshape(-1,)  # Reshape, 将y_pred 变为1维度
acc = np.sum((y_pred-y_test)==0)/len(y_test)
print(f"acc={acc:.3f}")

#%% 画出SVM分类情况
plt.figure()
plt.clf()
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, zorder=500, cmap=plt.cm.Paired,
            edgecolor='k', s=20)

plt.axis('tight')
x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()
y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

plt.show()

