
# 导入包
import nibabel as nib
import numpy as np
import os   # 专门处理路径，文件名
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from eslearn.model_evaluator import ModelEvaluator

#%% 读取一个人的nifti数据
example_file = r"F:\workshop\demo_data\piVShc_FCS\HC\zDegreeCentrality_PositiveWeightedSumBrainMap_sub047.nii"
obj = nib.load(example_file)
affine = obj.affine
data = obj.get_fdata()
data = np.reshape(data, [-1,])  # 拉直

#%% 读取一组人的数据
def load_data(root):
    file_name = os.listdir(root)  # 把root的下所有文件名列出了
    file_path = [os.path.join(root, fn) for fn in file_name]
    data = [nib.load(fp).get_fdata() for fp in file_path]
    data_2d = [np.reshape(dd, [-1,]) for dd in data] 
    data_2d = np.array(data_2d)
    mask = np.sum(np.abs(data_2d), axis=0)
    mask = mask != 0
    data_2d_nonezero = data_2d[:, mask]
    return data_2d_nonezero, mask

root_hc = r"F:\workshop\demo_data\piVShc_FCS\HC"
root_pi = r"F:\workshop\demo_data\piVShc_FCS\PI"

data_hc, mask_hc = load_data(root_hc)
data_pi, mask_pi = load_data(root_pi)

#%% 把病人和正常人的数据拼接成x
data = np.vstack([data_pi, data_hc])
label = np.hstack([np.ones([59,]), np.zeros(47,)])

#%% 数据检查

# 缺失值检查
nan_value = np.isnan(data)
nan_acount = np.sum(nan_value, axis=0)
np.sum(nan_acount)

# 极端值检查
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_z = (data-data_mean)/data_std
extra_mask = np.abs(data_z) > 3
extra_acount = np.sum(extra_mask, axis=0)
plt.hist(extra_acount)

# 拆分数据集
accuracy = []
skf = StratifiedKFold(n_splits=2)
for train_index, test_index in skf.split(data, label):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 把训练集的极端值填充
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train_z = (x_train-x_train_mean)/x_train_std
    extra_mask = np.abs(x_train_z) > 3
    extra_acount = np.sum(extra_mask, axis=0)
    plt.hist(extra_acount)
    x_train[extra_mask] = np.nan
    x_train = pd.DataFrame(x_train)
    mean_value = x_train.mean()
    x_train.fillna(value=mean_value, inplace=True)
    
    # 要用训练集的均数填充测试集
    x_test_z = (x_test-x_train_mean)/x_train_std
    extra_mask = np.abs(x_test_z) > 3
    extra_acount = np.sum(extra_mask, axis=0)
    x_test[extra_mask] = np.nan
    
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
    ros = RandomOverSampler(random_state=0)
    x_train, y_train = ros.fit_sample(x_train, y_train)
    ul = np.unique(y_train)
    negl = ul[0]
    posl = ul[1]
    print(np.sum(y_train==negl), np.sum(y_train==posl))
    
    def get_cv_metric(x_train, y_train, searchspace):
        
        method_scaler=searchspace["method_scaler"]
        n_cmp = searchspace["n_cmp"]
        method_est = searchspace["method_est"]
        
        # 数据规范化：z标准化或者归一化
        scaler = method_scaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        
        # 降维
        dr = PCA(n_components=n_cmp, random_state=0)
        dr.fit(x_train)
        x_train = dr.transform(x_train)
        
        # 特征选择
        lsvc = method_est()
        selector = RFECV(estimator=lsvc, step=0.1)
        selector.fit(x_train, y_train)
        x_train = selector.transform(x_train)
        
        # 交叉验证调整参数
        lsvc = LinearSVC(C=1, random_state=0)
        cv_metric = cross_val_score(lsvc, x_train, y_train)
        mean_metric = np.mean(cv_metric)
        return mean_metric
        
    def f(searchspace):
        acc = get_cv_metric(x_train, y_train, searchspace)
        return {'loss': -acc, 'status': STATUS_OK}
    
    searchspace = {
        "method_scaler": hp.choice("ms", [StandardScaler]),
        "n_cmp": hp.uniform("ncmp", 0.7, 0.99),
        "method_est":hp.choice("mest", [LogisticRegression, RidgeClassifier]),
    }
    
    trials = Trials()
    best = fmin(f, searchspace, algo=tpe.suggest, max_evals=5, trials=trials)
    print(best)
    
    # 数据规范化：z标准化或者归一化
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    # 降维
    dr = PCA(n_components=0.9, random_state=0)
    dr.fit(x_train)
    x_train = dr.transform(x_train)
    x_test = dr.transform(x_test)
    
    # 特征选择
    lsvc = LinearSVC()
    selector = RFECV(estimator=lsvc, step=0.1)
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    
    # 交叉验证调整参数
    lsvc = LinearSVC(C=1, random_state=0)
    cv_metric = cross_val_score(lsvc, x_train, y_train)
    mean_metric = np.mean(cv_metric)
    print(mean_metric)
    
    # 建模
    model =  LinearSVC(C=1, random_state=0)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    model.score(x_train, y_train)
    
    # 获得特征的权重
    weight = model.coef_
    
    # 测试
    acc = model.score(x_test, y_test)
    pred_test = model.predict(x_test)
    accuracy.append(acc)
    
    # 获得预测的概率，或者预测的决策函数值
    if hasattr(model, "decision_function"):
        pred_prob = model.decision_function(x_test)
    elif hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(x_test)
        
mean_accuracy = np.mean(accuracy)

# 作图
meval = ModelEvaluator()
meval.binary_evaluator(true_label=y_test, 
                       predict_label=pred_test, 
                       predict_score=pred_prob,
                       accuracy_kfold=None,   # 把None改为真实值
                       sensitivity_kfold=None, 
                       specificity_kfold=None, 
                       AUC_kfold=None,
                       verbose=True, 
                       is_showfig=True, 
                       legend1='HC', 
                       legend2='Patients', 
                       is_savefig=True, 
                       out_name=r"./fig.pdf")

# ROC 曲线








