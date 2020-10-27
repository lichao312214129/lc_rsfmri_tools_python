# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:22:59 2020
Preprocessing data

NOTE. After analysis, "CURVATURE", and "TORSION" should exclude due to their high nan ratio
@author: Li Chao
"""

import scipy.io as sio
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd


METRICS = ["FA", "MD", "RD", "AD", "CL", "VOLUME"]


def preprocess(file, name="train_set", num_sub=700, num_fiber=20):
    """ Preprocess data
    """

    data_struct = sio.loadmat(file)  
    feature = data_struct[name]
    
    # all_num_an = {}
    # all_proportion_na = []
    # for metrics_ in METRICS:
    #     proportion_na = []
    #     for i, feature_ in enumerate(feature):
    #         data = feature_[metrics_][0]
    #         na = np.sum(np.isnan(data), axis=1) > 0
    #         if i == 0:
    #             n_sub = len(data)
    #             num_na = np.zeros([n_sub,], dtype=np.int32)
    #         num_na += np.int32(na)  # Number of nan of each subject
    #         proportion_na.append(np.sum(na)/n_sub)  # Proportion of nan of each fiber
    #         print(f"{i}: {np.sum(na)/n_sub:.4f}")
        
    #     all_num_an[metrics_] = num_na
    #     all_proportion_na.append(proportion_na)
    # all_proportion_na = pd.DataFrame(all_proportion_na).T
    # all_proportion_na.columns = METRICS
    
    # Fibers index the needed to drop are 4,6,7
    data_for_all = {}
    num_na = np.zeros([num_sub,], dtype=np.int32)
    for metrics_ in METRICS:
        for i, feature_ in enumerate(feature):
            data = feature_[metrics_][0]
            if i == 0:
                data_for_one_metric = data
            elif i not in (4,6,7):
                data_for_one_metric = np.concatenate([data_for_one_metric, data], axis=1) 
            
            # Mean number of nan of each subject across all metrics
            # I make this is a feature
            na = np.sum(np.isnan(data), axis=1) > 0
            num_na += np.int32(na)  # Number of nan of each subject
            
        data_for_all[metrics_] = data_for_one_metric
        mean_num_na = num_na/len(METRICS)
        
    return data_for_all, mean_num_na

if __name__ == "__main__":
    file = r'F:\AD分类比赛\MCAD_AFQ_competition.mat'
    data_struct = preprocess(file)
    feature = data_struct["train_set"]
    
    np.mean(mean_num_na[np.reshape(data_struct["train_diagnose"]==1, [-1,])])
    np.mean(mean_num_na[np.reshape(data_struct["train_diagnose"]==2, [-1,])])
    np.mean(mean_num_na[np.reshape(data_struct["train_diagnose"]==3, [-1,])])
