# -*- coding: utf-8 -*-
""" Identify subjects index with repeated scans

Created on Thu May 21 10:31:37 2020

@author: Li Chao
"""

import sys
sys.path.append(r'D:\My_Codes\lc_private_codes\utils')
import pandas as pd
import numpy as np
from lc_indentify_repeat_subjects import indentify_repeat_subjects_pairs as irsp

# Inputs
scale_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\10-24大表.xlsx'
discovery_idx_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\id.xlsx'
validation_idx_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\held_out_samples.txt'

# Load
scale = pd.read_excel(scale_file)
discovery_idx = pd.read_excel(discovery_idx_file, header=None)
validation_idx = pd.read_csv(validation_idx_file, header=None)

# Identity repeated id
repeat_second, repeat_first = irsp(
    file=r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\10-24大表.xlsx',
    uid_header='folder'
)

# Make folder and id to int
repeat_second_int = pd.DataFrame([int(rp[0]) if rp != [] else 0 for rp in repeat_second])
repeatPairs = np.concatenate([repeat_first.values.reshape(-1,1), repeat_second_int.values.reshape(-1,1)], axis=1)
for i in range(repeatPairs.shape[0]):
    if repeatPairs[i, 0] > repeatPairs[i, 1]:
        repeatPairs[i, 0], repeatPairs[i, 1] = repeatPairs[i, 1], repeatPairs[i, 0]

# Delete repeated uncertain subjects and only retained subjects with twice scans
repeatPairs = pd.DataFrame(repeatPairs)
du = repeatPairs[1].drop_duplicates()
repeatPairs_only_twice = repeatPairs.loc[du.index,:]
du = repeatPairs_only_twice[1] != 0
repeatPairs_only_twice = repeatPairs_only_twice.loc[du,:]

du = repeatPairs[0].drop_duplicates()
repeatPairs_only_twice = repeatPairs.loc[du.index,:]
du = repeatPairs_only_twice[0] != 0
repeatPairs_only_twice = repeatPairs_only_twice.loc[du,:]

# Sort repeatPairs
repeatPairs_only_twice=repeatPairs_only_twice.sort_values(by=0)

# Identify repeated subjects in discovery and validation datasets
repeat_id_in_discovery_first_column = pd.merge(discovery_idx, repeatPairs_only_twice[0], left_on=0, right_on=0, how='inner')
repeat_id_in_discovery_second_column = pd.merge(discovery_idx, repeatPairs_only_twice[1], left_on=0, right_on=1, how='inner')

repeat_id_in_validation_first_column = pd.merge(validation_idx, repeatPairs_only_twice[0], left_on=0, right_on=0, how='inner')
repeat_id_in_validation_second_column = pd.merge(validation_idx, repeatPairs_only_twice[1], left_on=0, right_on=1, how='inner')

cov = np.in1d(repeat_id_in_validation_first_column, repeat_id_in_discovery_first_column)

# Identify subjects with diagnosis changed
diagnosis_first = pd.merge(repeatPairs_only_twice[0],scale, left_on=0, right_on='folder', how='left')[['folder', '诊断']]
diagnosis_second = pd.merge( repeatPairs_only_twice[1], scale, left_on=1, right_on='folder', how='left')[['folder', '诊断']]

diagnosis_changed_loc = (diagnosis_first['诊断'] - diagnosis_second['诊断']) != 0

changed_before = diagnosis_first.iloc[diagnosis_changed_loc.values, :]
changed_after = diagnosis_second.iloc[diagnosis_changed_loc.values, :]

# 
loc_repeated_paire_in_discovery = np.in1d(repeatPairs_only_twice[0], repeat_id_in_discovery_first_column.values)
repeated_paire_in_discovery = repeatPairs_only_twice.iloc[loc_repeated_paire_in_discovery,:]

a=np.in1d(diagnosis_first.index[diagnosis_first['诊断']==6], diagosis_second.index[diagnosis_second['诊断']==6])