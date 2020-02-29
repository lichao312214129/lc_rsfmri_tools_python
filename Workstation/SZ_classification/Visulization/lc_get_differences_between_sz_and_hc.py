# -*- coding: utf-8 -*-
"""
This script is used to get the difference between SZ and HC.
These differences including all SZ vs HC, and first episode unmedicated SZ vc matching HC.

NOTO. This part of script is only used to preprocess the data, 
then submit the data to MATLAB to add weight mask and visualization.
"""

import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Workstation\SZ_classification\ML')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Statistics')
import numpy as np
import  pandas as pd
from lc_pca_svc_pooling import PCASVCPooling
import scipy.io as sio
from scipy.stats import ttest_ind

from lc_calc_cohen_d_effective_size import CohenEffectSize

# Unique index of first episode unmedicated patients
dataset_first_episode_unmedicated_path = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_unmedicated_and_firstepisode_550.npy'
scale = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'

#%% Load all dataset
scale = pd.read_excel(scale)
sel = PCASVCPooling()
dataset_our_center_550 = np.load(sel.dataset_our_center_550)
dataset_206 = np.load(sel.dataset_206)
dataset_COBRE = np.load(sel.dataset_COBRE)
dataset_UCAL = np.load(sel.dataset_UCAL)
dataset1_firstepisodeunmed = np.load(dataset_first_episode_unmedicated_path)

# Extract features and label
features_our_center_550 = dataset_our_center_550[:, 2:]
features_206 = dataset_206[:, 2:]
features_COBRE = dataset_COBRE[:, 2:]
features_UCAL = dataset_UCAL[:, 2:]
feature_firstepisodeunmed = dataset1_firstepisodeunmed[:, 2:]

label_our_center_550 = dataset_our_center_550[:, 1]
label_206 = dataset_206[:, 1]
label_COBRE = dataset_COBRE[:, 1]
label_UCAL = dataset_UCAL[:, 1]
label_firstepisodeunmed = dataset1_firstepisodeunmed[:, 1]

#%% Generate training data and test data
# All
data_all = np.concatenate(
    [features_our_center_550, features_206, features_UCAL, features_COBRE], axis=0)
label_all = np.concatenate(
    [label_our_center_550, label_206, label_UCAL, label_COBRE], axis=0)


#%% Get the difference
# All
data_sz = data_all[label_all == 1]
data_hc = data_all[label_all == 0]

# [differences_all, _] = ttest_ind(data_sz, data_hc)
cohen_diff_all = CohenEffectSize(data_sz, data_hc)

# First episode unmedicated
data_sz_firstepisodeunmed = feature_firstepisodeunmed[label_firstepisodeunmed == 1]
data_hc_firstepisodeunmed = feature_firstepisodeunmed[label_firstepisodeunmed == 0]
# [differences_feu, _] = ttest_ind(data_sz_firstepisodeunmed, data_hc_firstepisodeunmed)
cohen_diff_feu = CohenEffectSize(data_sz_firstepisodeunmed, data_hc_firstepisodeunmed)

#%% Make the differences to 2D matrix and save to mat
cohen_all = np.zeros([246,246])
cohen_all[np.triu(np.ones([246,246]), 1) == 1] = cohen_diff_all

cohen_feu = np.zeros([246,246])
cohen_feu[np.triu(np.ones([246,246]), 1) == 1] = cohen_diff_feu

sio.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\differecens_all.mat', {'differences_all': cohen_all})
sio.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\differecens_feu.mat', {'differences_feu': cohen_feu})
