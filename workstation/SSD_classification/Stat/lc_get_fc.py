# -*- coding: utf-8 -*-
""" This script is used to save fc to .mat file for NBS in MATLAB:

"""

import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from lc_calc_cohen_d_effective_size import CohenEffectSize

# Inputs
is_save = 1
dataset1_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy'
cov_chronic = r'D:\WorkStation_2018\SZ_classification\Scale\cov_chronic.txt'
cov_firstepisode_medicated = r'D:\WorkStation_2018\SZ_classification\Scale\cov_firstepisode_medicated.txt'
cov_firstepisode_unmedicated = r'D:\WorkStation_2018\SZ_classification\Scale\cov_firstepisode_unmedicated.txt'

# Load all dataset
dataset = np.load(dataset1_file)
dataset_our_center_550 = np.load(sel.dataset_our_center_550)

cov_chronic = pd.read_csv(cov_chronic)
cov_firstepisode_medicated = pd.read_csv(cov_firstepisode_medicated)
cov_firstepisode_unmedicated =pd.read_csv(cov_firstepisode_unmedicated)

# Extract corr data
merged_data_cov_chronic = pd.merge(pd.DataFrame(dataset), cov_chronic, left_on=0, right_on='folder')
merged_data_cov_firstepisode_medicated = pd.merge(pd.DataFrame(dataset), cov_firstepisode_medicated, left_on=0, right_on='folder')
merged_data_cov_firstepisode_unmedicated = pd.merge(pd.DataFrame(dataset), cov_firstepisode_unmedicated, left_on=0, right_on='folder')

fc_chronic = merged_data_cov_chronic.iloc[:, np.arange(0, np.shape(dataset)[1])].drop([0,1], axis=1)
fc_firstepisode_medicated = merged_data_cov_firstepisode_medicated.iloc[:, np.arange(0, np.shape(dataset)[1])].drop([0,1], axis=1)
fc_firstepisode_unmedicated = merged_data_cov_firstepisode_unmedicated.iloc[:, np.arange(0, np.shape(dataset)[1])].drop([0,1], axis=1)

cov_chronic = merged_data_cov_chronic.iloc[:, np.arange(-1, -np.shape(cov_chronic)[1]-1, -1)]
cov_firstepisode_medicated = merged_data_cov_firstepisode_medicated.iloc[:, np.arange(-1, -np.shape(cov_firstepisode_medicated)[1]-1, -1)]
cov_firstepisode_unmedicated = merged_data_cov_firstepisode_unmedicated.iloc[:, np.arange(-1, -np.shape(cov_firstepisode_unmedicated)[1]-1, -1)]

# Calc cohen
cohen_duration = CohenEffectSize(fc_chronic, fc_firstepisode_medicated)
cohen_medication = CohenEffectSize(fc_firstepisode_medicated, fc_firstepisode_unmedicated)

# Make the differences to 2D matrix and save to mat
cohen_duration_full = np.zeros([246,246])
cohen_duration_full[np.triu(np.ones([246,246]), 1) == 1] = cohen_duration
cohen_duration_full = cohen_duration_full + cohen_duration_full.T

cohen_medication_full = np.zeros([246,246])
cohen_medication_full[np.triu(np.ones([246,246]), 1) == 1] = cohen_medication
cohen_medication_full = cohen_medication_full + cohen_medication_full.T

#%% Save to mat for MATLAB process (NBS)
if is_save:
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\cohen_duration.mat', {'cohen_duration': cohen_duration_full})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\cohen_medication.mat', {'cohen_medication': cohen_medication_full})

    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\fc_chronic.mat', {'fc_chronic': fc_chronic})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\fc_firstepisode_medicated.mat', {'fc_firstepisode_medicated': fc_firstepisode_medicated})
    sio.savemat(r'D:\WorkStation_2018\SZ_classification\Data\fc_firstepisode_unmedicated.mat', {'fc_firstepisode_unmedicated': fc_firstepisode_medicated})
