# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:57:52 2020
Split the data into training, validation and test sets
@author: Li Chao, Dong Mengshi
"""

from sklearn.model_selection  import StratifiedShuffleSplit
import scipy.io as sio
import numpy as np
from preprocess import preprocess
import pandas as pd
import matplotlib.pyplot as plt

def split(seed=66):
    # Input
    n_sub = 700
    file = r'F:\AD分类比赛\MCAD_AFQ_competition.mat'
    trp = 0.6
    vp = 0.2
    tep = 0.2

    # Load
    data_struct = sio.loadmat(file)  
    label = data_struct["train_diagnose"]
    site = data_struct["train_sites"]
    data_for_all, label = preprocess(file)

    # Get training , validation and test sets
    data_for_all = {key: data_for_all[key] for key in data_for_all.keys()}

    # Get index for training, validation and test sets
    np.random.seed(seed)
    
    # skf = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=666)
    # skf_index = list(skf.split(data_for_all["AD"], label))
    # idx_train = skf_index[0][0]
    
    # data_train_ad = data_for_all["AD"][skf_index[0][0],:]
    # label_train = label[skf_index[0][0]]
    # feature_other = data_for_all["AD"][skf_index[0][1],:]
    # label_other = label[skf_index[0][1]]
    
    # skf = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=666)
    # vt_index = list(skf.split(feature_other, label_other))
    # idx_validation = vt_index[0][0]
    # idx_test = vt_index[0][1]
    
    
    idx = np.random.permutation(n_sub)
    idx_train = idx[:int(n_sub*trp)]
    idx_validation = idx[int(n_sub*trp):int(n_sub*(trp+vp))]
    idx_test = idx[int(n_sub*(trp+vp)):]

    # Check site distribution
    site_train = pd.DataFrame(site[idx_train])
    site_validation = pd.DataFrame(site[idx_validation])
    site_test = pd.DataFrame(site[idx_test])
    count_train = site_train[0].value_counts()
    count_validation = site_validation[0].value_counts()
    count_test = site_test[0].value_counts()
    ratio_train = [count_train.loc[i]/count_train.sum() for i in np.arange(1,8)]
    ratio_validation = [count_validation.loc[i]/count_validation.sum() for i in np.arange(1,8)]
    ratio_test = [count_test.loc[i]/count_test.sum() for i in np.arange(1,8)]
    
    # plt.plot(ratio_train)
    # plt.plot(ratio_validation)
    # plt.plot(ratio_test)
    # plt.legend(["Tr","Val","Te"])
    # plt.show()
    # Get training , validation and test sets
    data_train = {key: data_for_all[key][idx_train,:] for key in data_for_all.keys()}
    data_validation = {key: data_for_all[key][idx_validation,:] for key in data_for_all.keys()}
    data_test = {key: data_for_all[key][idx_test,:] for key in data_for_all.keys()}

    # mean_num_na_train = mean_num_na[idx_train]
    # mean_num_na_validation = mean_num_na[idx_validation]
    # mean_num_na_test = mean_num_na[idx_test]

    # demo_train = demo[idx_train]
    # demo_validation = demo[idx_validation]
    # demo_test = demo[idx_test]

    label_train = label[idx_train,:].reshape(-1,)
    label_validation = label[idx_validation,:].reshape(-1,)
    label_test = label[idx_test,:].reshape(-1,)

    return (data_train, data_validation, data_test, label_train, label_validation, label_test)


