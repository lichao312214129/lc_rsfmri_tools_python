# 导入包
import nibabel as nib
import numpy as np
import os   # 专门处理路径，文件名
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso
from sklearn.model_selection import cross_val_score
from eslearn.model_evaluator import ModelEvaluator

meval = ModelEvaluator()

#%% ==============================TRAINING=====================================
#%% 读取一个人的nifti数据
example_file = r"E:\Mechine_studying\demo_data\depression\data.xlsx"
data_all = pd.read_excel(example_file)

# 获取x和y
data = data_all.iloc[:, 2:]  # 因为是dataframe的格式，所以我们要用.iloc来切片
label = data_all.iloc[:, 1]

feature_name = list(data.columns)

# 把dataframe格式的数据变成numpy.ndarray
data = data.values
label = label.values

#%% 数据检查

# 缺失值检查
nan_value = np.isnan(data)
nan_acount = np.sum(nan_value, axis=0)
np.sum(nan_acount)

# # 极端值检查
# data_mean = np.mean(data, axis=0)
# data_std = np.std(data, axis=0)
# data_z = (data-data_mean)/data_std
# extra_mask = np.abs(data_z) > 3
# extra_acount = np.sum(extra_mask, axis=0)

# 划分数据集训练：验证：测试集=6:2:2
np.random.seed(666)  # 固定随机种子点，保证每次划分一致
n_cases = len(label)  # 表示长度
idx = np.random.permutation(n_cases)  # 生成随机数列

idx_train = idx[0:np.int(n_cases*0.6), ]  # 切片，取前面60%，并转化为整数类型，即四舍五入
idx_validation = idx[np.int(n_cases*0.6):np.int(n_cases*0.8), ]  # 取中间20%
idx_test = idx[np.int(n_cases*0.8):, ]  # 取后面20%

x_train = data[idx_train,:]  # “：”表示取所有列
x_validation = data[idx_validation,:]
x_test = data[idx_test,:]

y_train = label[idx_train]
y_validation = label[idx_validation]
y_test = label[idx_test]

# 把全部人相等的特征去除
unique_data = [len(np.unique(x_train[:,i])) for i in range(np.shape(x_train)[1])]
mask_include = [unique_data_ != 1 for unique_data_ in unique_data]
x_train = x_train[:, mask_include]
x_validation = x_validation[:, mask_include]
x_test = x_test[:, mask_include]

# 把训练集的极端值去除
x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train_z = (x_train-x_train_mean)/x_train_std
extra_mask = np.abs(x_train_z) > 3
if len(x_train[extra_mask]) > 0:
    x_train[extra_mask] = np.nan
    
x_train = pd.DataFrame(x_train)
mean_value = x_train.mean()
x_train.fillna(value=mean_value, inplace=True)

# 要用训练集的均数填充测试集
x_test = pd.DataFrame(x_test)
x_test.fillna(value=mean_value, inplace=True)

# 把DataFrame变成ndarray
x_train = x_train.values
x_test = x_test.values

# 平衡类别之间的样本量
ul = np.unique(y_train)
negl = ul[0]
posl = ul[1]
print(np.sum(y_train==negl), np.sum(y_train==posl))
ros = RandomOverSampler(random_state=0)  # 上采样
x_train, y_train = ros.fit_sample(x_train, y_train)
ul = np.unique(y_train)
negl = ul[0]
posl = ul[1]
print(np.sum(y_train==negl), np.sum(y_train==posl))

# 数据规范化：z标准化或者归一化
scaler = StandardScaler()  # 实例化
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# # 降维
# dr = PCA(n_components=0.9,random_state = 0)  # PCA降维  可以用其他降维方式
# dr.fit(x_train)
# x_train = dr.transform(x_train)
# x_test = dr.transform(x_test)

# 特征选择
lsvc = LinearSVC()
selector = RFECV(estimator=lsvc, step=1)  # RFECV递归特征消除
selector.fit(x_train, y_train)
x_train = selector.transform(x_train)  # transform表示应用到...上去
x_test = selector.transform(x_test)

# 交叉验证调整参数
lsvc = LinearSVC(C=1, random_state=0)
cv_metric = cross_val_score(lsvc, x_train, y_train) # cross_val_score表示交叉验证的方法
mean_metric = np.mean(cv_metric)


# 建模
# model =  LinearSVC(C=1, random_state=0)  # LinearSVC是线性支持向量机
model = SVC(C=1.0, kernel='poly', random_state=0)
model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_train, y_train)

# 获得特征的权重
weight = model.coef_

# 把系数投射到原始的空间，或者维度
weight = selector.inverse_transform(weight)

weight_ = np.zeros(np.shape(mask_include))
weight = np.reshape(weight, [-1, ])
weight_[mask_include] = weight

#%% 测试
model.score(x_test, y_test)
pred_test = model.predict(x_test)

# 获得预测的概率，或者预测的决策函数值
if hasattr(model, "decision_function"):
    pred_prob = model.decision_function(x_test)
elif hasattr(model, "predict_proba"):
    pred_prob = model.predict_proba(x_test)


# model eval
acc, sen, spec, auc, _ = \
    meval.binary_evaluator(true_label=y_test, 
                       predict_label=pred_test, 
                       predict_score=pred_prob,
                       verbose=True, 
                       is_showfig=False, 
                       is_savefig=False)


accuracy_kfold.append(acc)
sensitivity_kfold.append(sen)
specificity_kfold.append(spec)
AUC_kfold.append(auc)
    
    
#%%
# 作图
meval = ModelEvaluator()
meval.binary_evaluator(true_label=y_test, 
                       predict_label=pred_test, 
                       predict_score=pred_prob,
                       accuracy_kfold=accuracy_kfold,   # 把None改为真实值
                       sensitivity_kfold=sensitivity_kfold, 
                       specificity_kfold=specificity_kfold, 
                       AUC_kfold=AUC_kfold,
                       verbose=True, 
                       is_showfig=True, 
                       legend1='HC', 
                       legend2='Patients', 
                       is_savefig=True, 
                       out_name=r"./fig.pdf")

















