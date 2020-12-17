# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:13:18 2020

@author: lenovo
"""

# 导入模块
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso
from sklearn.metrics import confusion_matrix

# 加载数据
data_file = '../demo_data/winequality-red.csv'
data = pd.read_csv(data_file, sep=';')

#把数据头名字改成中文
header = ["固定酸度","挥发性酸度","柠檬酸","残糖","氯化物","游离二氧化硫","总二氧化硫","密度","PH","硫酸盐","酒精", "质量"]
data.columns = header

# 检查数据
describe = data.describe()
data.info()

# 获取数据的值
data_values = data.values

# 检查数据是否有缺失值
nan_values = np.isnan(data_values)
nan_sum = np.sum(nan_values, axis=0)

# 检查相关性
plt.figure()
coef = spearmanr(data)
coef = coef[0]
sns.heatmap(coef, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, square=True, fmt=".2f")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xticks(np.arange(0,12)+0.5, header, rotation=90)
plt.yticks(np.arange(0,12)+0.5, header, rotation=0)
plt.tight_layout()
plt.show()

# 获取数据的特征和targets
features = data_values[:, 0:11]
label = data_values[:, -1]

# 查看有几个类别
np.unique(label)

# 我们把3,4设置为第一个类别，5，6设置为第二个类别，7，8设置为第三个类别
label[np.in1d(label, [3, 4, 5])] = 0 
label[np.in1d(label, [6, 7, 8])] = 1

np.sum(label==0)
np.sum(label==1)

np.unique(label)

# 划分数据集训练：验证：测试集=6:2:2
np.random.seed(666)
n_cases = len(features)
idx = np.random.permutation(n_cases)

idx_train = idx[0:np.int(n_cases*0.6), ]
idx_validation = idx[np.int(n_cases*0.6):np.int(n_cases*0.8), ]
idx_test = idx[np.int(n_cases*0.8):, ]

feature_train = features[idx_train,:]
feature_validation = features[idx_validation,:]
feature_test = features[idx_test,:]

label_train = label[idx_train]
label_validation = label[idx_validation]
label_test = label[idx_test]

# 平衡数据
print(f"Before re-sampling, the sample size are: {sorted(Counter(label_train).items())}")
feature_train, label_train = RandomOverSampler().fit_resample(feature_train, label_train)
print(f"After re-sampling, the sample size are: {sorted(Counter(label_train).items())}")

# 数据规范化:zscore
scaler = StandardScaler()
scaler.fit(feature_train)

feature_train = scaler.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = scaler.transform(feature_validation)
feature_test = scaler.transform(feature_test)
 
# 特征工程：降维和筛选（可选）

# pca = PCA(n_components=0.9)
# pca.fit(feature_train)
# feature_train = pca.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
# feature_validation = pca.transform(feature_validation)
# feature_test = pca.transform(feature_test)

plf =PolynomialFeatures(3)
plf.fit(feature_train)
feature_train = plf.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = plf.transform(feature_validation)
feature_test = plf.transform(feature_test)

# 特征筛选
# selector = Lasso(alpha=0.07)
# selector.fit(feature_train, label_train)
# mask = selector.coef_ != 0
# feature_train = feature_train[:, mask] # 切记：要使用训练集的参数对验证集和测试集进行处理
# feature_validation = feature_validation[:,mask]
# feature_test = feature_test[:, mask]

selector = RFECV(estimator=LinearSVC(), step=.1)
selector.fit(feature_train, label_train)
feature_train = selector.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = selector.transform(feature_validation)
feature_test = selector.transform(feature_test)

# 建模
model = LogisticRegression()
model.fit(feature_train, label_train)

acc_train = model.score(feature_train, label_train)
acc_validation = model.score(feature_validation, label_validation)
print(acc_train, acc_validation)

# 测试
# pred_test = model.predict(feature_test)

# acc = np.sum((pred_test - label_test) == 0)/len(pred_test)
# print(acc)

# plt.figure()
# cm = confusion_matrix(label_test, pred_test)
# sns.heatmap(cm, cmap="RdBu_r", annot=True, square=True, fmt=".2f")
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # plt.xticks(np.arange(0,12)+0.5, header, rotation=90)
# # plt.yticks(np.arange(0,12)+0.5, header, rotation=0)
# plt.tight_layout()
# plt.show()