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
features_file = '../demo_data/features_whitewine.csv'
targets_file = '../demo_data/targets_whitewine.csv'
features = pd.read_csv(features_file)
targets = pd.read_csv(targets_file)

# 查看表头
header = features.columns

# 检查数据
describe = features.describe()
features.info()

# 获取数据的值
features = features.values[:, 1:]
targets = targets["__Targets__"].values

# 检查数据是否有缺失值
nan_values = np.isnan(features)
nan_sum = np.sum(nan_values, axis=0)

# 查看有几个类别
np.unique(targets)

# 两个类别各有多少cases
np.sum(targets==0)
np.sum(targets==1)

# 划分数据集训练：验证：测试集=6:2:2
np.random.seed(666)
n_cases = len(targets)
idx = np.random.permutation(n_cases)

idx_train = idx[0:np.int(n_cases*0.6), ]
idx_validation = idx[np.int(n_cases*0.6):np.int(n_cases*0.8), ]
idx_test = idx[np.int(n_cases*0.8):, ]

feature_train = features[idx_train,:]
feature_validation = features[idx_validation,:]
feature_test = features[idx_test,:]

targets_train = targets[idx_train]
targets_validation = targets[idx_validation]
targets_test = targets[idx_test]

# 平衡数据
print(f"Before re-sampling, the sample size are: {sorted(Counter(targets_train).items())}")
rs = RandomOverSampler()
feature_train, targets_train = rs.fit_resample(feature_train, targets_train)
print(f"After re-sampling, the sample size are: {sorted(Counter(targets_train).items())}")

# 数据规范化:zscore
scaler = StandardScaler()
scaler.fit(feature_train)

feature_train = scaler.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = scaler.transform(feature_validation)
feature_test = scaler.transform(feature_test)
 
# 特征工程：降维和筛选(可选)
# pca = PCA(n_components=0.9, random_state=666)
# pca.fit(feature_train)
# feature_train = pca.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
# feature_validation = pca.transform(feature_validation)
# feature_test = pca.transform(feature_test)

# 特征
plf = PolynomialFeatures(3)
plf.fit(feature_train)
feature_train = plf.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = plf.transform(feature_validation)
feature_test = plf.transform(feature_test)

# 特征筛选1
selector = Lasso(alpha=0.02)
selector.fit(feature_train, targets_train)
mask = selector.coef_ != 0
feature_train = feature_train[:, mask] # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = feature_validation[:,mask]
feature_test = feature_test[:, mask]

# 特征筛选2
# selector = RFECV(estimator=LinearSVC(), cv=3, step=0.1)
# selector.fit(feature_train, targets_train)
# feature_train = selector.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
# feature_validation = selector.transform(feature_validation)
# feature_test = selector.transform(feature_test)

# 建模
model = LinearSVC()
model.fit(feature_train, targets_train)
coef = model.coef_

acc_train = model.score(feature_train, targets_train)
acc_validation = model.score(feature_validation, targets_validation)
print(acc_train, acc_validation)

# 测试
pred_test = model.predict(feature_test)
acc = np.sum((pred_test - targets_test) == 0)/len(pred_test)
print(acc)

# 敏感度
idx = [np.bool(tp) for tp in targets_test]
pt = pred_test[idx]
fenzi = np.sum(pt)
fenmu = np.sum(targets_test == 1)
sen = fenzi/fenmu

# 特异度