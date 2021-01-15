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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso
from sklearn.metrics import confusion_matrix

# 加载数据
features_file = r'D:\My_Codes\lc_private_codes\The_first_ml_training\demo_data/features_whitewine.csv'
targets_file = r'D:\My_Codes\lc_private_codes\The_first_ml_training\demo_data/targets_whitewine.csv'
features = pd.read_csv(features_file)
targets = pd.read_csv(targets_file)

# 查看表头
header = features.columns

# 检查数据
describe = features.describe()
features.info()

# 获取数据的值
features = features.values[:,1:]
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
# print(f"Before re-sampling, the sample size are: {sorted(Counter(targets_train).items())}")
# feature_train, targets_train = RandomOverSampler().fit_resample(feature_train, targets_train)
# print(f"After re-sampling, the sample size are: {sorted(Counter(targets_train).items())}")

# 数据规范化:zscore
scaler = StandardScaler()
scaler.fit(feature_train)

feature_train = scaler.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
feature_validation = scaler.transform(feature_validation)
feature_test = scaler.transform(feature_test)


def get_model_acc(kwargs, 
                  feature_train, feature_validation, 
                  targets_train, targets_validation):
    """获取模型的准确度
    """
    
    # 解析输入参数
    # n_components = kwargs["n_components"]
    # kwargs = {"degree":1, "alpha": 0.01}
    degree = kwargs["degree"]
    alpha = kwargs["alpha"]
    
    # 降维
    # pca = PCA(n_components=n_components, random_state=666)
    # pca.fit(feature_train)
    # feature_train = pca.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
    # feature_validation = pca.transform(feature_validation)

    # 生成多项式特征
    plf = PolynomialFeatures(degree=degree)
    plf.fit(feature_train)
    feature_train = plf.transform(feature_train) # 切记：要使用训练集的参数对验证集和测试集进行处理
    feature_validation = plf.transform(feature_validation)
    
    # 特征筛选1
    selector = Lasso(alpha=alpha)
    selector.fit(feature_train, targets_train)
    mask = selector.coef_ != 0
    feature_train = feature_train[:, mask] # 切记：要使用训练集的参数对验证集和测试集进行处理
    feature_validation = feature_validation[:,mask]
    
    # 建模
    model = LinearSVC()
    model.fit(feature_train, targets_train)
    
    # 获取准确度
    # acc_train = model.score(feature_train, targets_train)
    acc_validation = model.score(feature_validation, targets_validation)
    return acc_validation

def f(searchspace):
    acc = get_model_acc(searchspace, 
                        feature_train, feature_validation, 
                        targets_train, targets_validation)
    return {'loss': 1/acc, 'status': STATUS_OK}

# def f(x):
#     return x**2
# 参数寻优
searchspace = {
    'degree': hp.randint("degree", 4) + 1,
    'alpha': hp.uniform("alpha", 0.01, 0.1),
}

# x_space = hp.uniform("x", -5, 5)


trials = Trials()
best = fmin(f, searchspace, algo=tpe.suggest, max_evals=200, trials=trials)

#%% 用最佳的参数训练模模型，并测试
# pca = PCA(n_components=best["n_components"], random_state=666)
# pca.fit(feature_train)
# feature_train = pca.transform(feature_train)
# feature_test = pca.transform(feature_test)

# 生成多项式特征
plf = PolynomialFeatures(degree=best["degree"])
plf.fit(feature_train)
feature_train = plf.transform(feature_train)
feature_validation = plf.transform(feature_validation)
feature_test = plf.transform(feature_test)

# 特征筛选
selector = Lasso(alpha=best["alpha"])
selector.fit(feature_train, targets_train)
mask = selector.coef_ != 0
feature_train = feature_train[:,mask]
feature_validation = feature_validation[:,mask]
feature_test = feature_test[:,mask]

# 建模
model = LinearSVC()
model.fit(feature_train, targets_train)

# 获取准确度
acc_train = model.score(feature_train, targets_train)
acc_validation = model.score(feature_validation, targets_validation)
acc_test = model.score(feature_test, targets_test)