# -*- coding: utf-8 -*-
""" Check demographics for this project, and identify subjects with repeated scans.

Subjects with repeated scans will be used to validate the reproducibility and the temporal stability of the HYDRA combined with functional connectivity.

Created on Thu May 21 10:31:37 2020

@author: Li Chao
"""


import sys
import pandas as pd
import numpy as np
import os
from lc_indentify_repeat_subjects import indentify_repeat_subjects_pairs as irsp


#%% Inputs
scale_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\10-24大表.xlsx'
fc_1322 = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\FC_1322'
headmotion_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\头动参数_1322.xlsx'
discovery_idx_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\id.xlsx'
validation_idx_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\held_out_samples.txt'

# Load
scale = pd.read_excel(scale_file)
id_discovery = pd.read_excel(discovery_idx_file, header=None)
id_validation = pd.read_csv(validation_idx_file, header=None)

#%% Identity repeated id
repeat_second, repeat_first = irsp(
    file=r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\10-24大表.xlsx',
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


#%% Identify repeated subjects in discovery and validation datasets
id_repeat_in_discovery_first_column = pd.merge(id_discovery, repeatPairs_only_twice[0], left_on=0, right_on=0, how='inner')
id_repeat_in_discovery_second_column = pd.merge(id_discovery, repeatPairs_only_twice[1], left_on=0, right_on=1, how='inner')
id_repeat_in_validation_first_column = pd.merge(id_validation, repeatPairs_only_twice[0], left_on=0, right_on=0, how='inner')
id_repeat_in_validation_second_column = pd.merge(id_validation, repeatPairs_only_twice[1], left_on=0, right_on=1, how='inner')
cov = np.in1d(id_repeat_in_validation_first_column, id_repeat_in_discovery_first_column)

loc_repeated_paire_in_discovery = np.in1d(repeatPairs_only_twice[0], id_repeat_in_discovery_first_column.values)
repeated_paire_in_discovery = repeatPairs_only_twice.iloc[loc_repeated_paire_in_discovery,:]

loc_repeated_paire_in_validation = np.in1d(repeatPairs_only_twice[0], id_repeat_in_validation_first_column.values)
repeated_paire_in_validation = repeatPairs_only_twice.iloc[loc_repeated_paire_in_validation,:]

# Validation set demo info
validation_demoinfo = pd.merge(id_validation, scale, left_on=0, right_on='folder', how='inner')
np.sum(validation_demoinfo['诊断'] == 1)
np.sum(validation_demoinfo['诊断'] == 2)
np.sum(validation_demoinfo['诊断'] == 3)
np.sum(validation_demoinfo['诊断'] == 4)

#%% fc_1322 and their demo info
fc_1322_file = os.listdir(fc_1322)
fc_1322_file = pd.Series(fc_1322_file)
id_fc_1322 = fc_1322_file.str.findall('[1-9]\d*')
id_fc_1322 = pd.DataFrame([np.int16(id) for id in id_fc_1322])
id_fc_1322_scale_intersection = pd.merge(id_fc_1322, scale, left_on=0, right_on='folder', how='inner')

# repeated subj with discovery set
id_repeat_with_discovery = pd.merge(id_fc_1322, repeated_paire_in_discovery[1], left_on=0, right_on=1, how='inner')[0]
scale_fc_1322_discovery_repeat = pd.merge(id_repeat_with_discovery, scale, left_on=0, right_on='folder', how='inner')[['folder', '年龄', '性别', '病程月', '诊断']]

# repeated subj with validation set
id_repeat_with_validation = pd.merge(id_fc_1322, repeated_paire_in_validation[1], left_on=0, right_on=1, how='inner')[0]
scale_fc_1322_validation_repeat = pd.merge(id_repeat_with_validation, scale, left_on=0, right_on='folder', how='inner')[['folder', '年龄', '性别', '病程月', '诊断']]

# demo
np.sum(scale_fc_1322_discovery_repeat['诊断'] == 1)
np.sum(scale_fc_1322_discovery_repeat['诊断'] == 2)
np.sum(scale_fc_1322_discovery_repeat['诊断'] == 3)
np.sum(scale_fc_1322_discovery_repeat['诊断'] == 4)

np.sum(scale_fc_1322_validation_repeat['诊断'] == 1)
np.sum(scale_fc_1322_validation_repeat['诊断'] == 2)
np.sum(scale_fc_1322_validation_repeat['诊断'] == 3)
np.sum(scale_fc_1322_validation_repeat['诊断'] == 4)

# check changed diagnosis
diagnosis_discovery_second = scale_fc_1322_discovery_repeat[['folder', '诊断']]
id_diagnosis_discovery_first = pd.merge(id_repeat_with_discovery, repeated_paire_in_discovery, left_on=0, right_on=1, how='inner')
diagnosis_discovery_first = pd.merge(id_diagnosis_discovery_first, scale, left_on='0_y', right_on='folder', how='inner')[['folder', '诊断']]
changed_diagnosis_of_discovery = diagnosis_discovery_second['诊断'] - diagnosis_discovery_first['诊断']  # No changed diagnosis

diagnosis_validation_second = scale_fc_1322_validation_repeat[['folder', '诊断']]
id_diagnosis_validation_first = pd.merge(id_repeat_with_validation, repeated_paire_in_validation, left_on=0, right_on=1, how='inner')
diagnosis_validation_first = pd.merge(id_diagnosis_validation_first, scale, left_on='0_y', right_on='folder', how='inner')[['folder', '诊断']]
changed_diagnosis_of_validation = diagnosis_validation_second['诊断'] - diagnosis_validation_first['诊断']  # No changed diagnosis

# Make sure on overlap between id_repeat_with_discovery and id_validation
overlap_id_fc_1322_repeat_and_id_validation = np.in1d(id_repeat_with_discovery, id_validation)

# Concat all repeated subjects from discovery and validation sets
id_repeated_with_discovery_and_validation = np.concatenate([id_repeat_with_discovery.values, id_repeat_with_validation.values])
changed_diagnosis_of_discovery_and_validation = np.concatenate([changed_diagnosis_of_discovery.values, changed_diagnosis_of_validation.values])
id_repeated_with_discovery_and_validation_with_no_change_diagnosis = id_repeated_with_discovery_and_validation[changed_diagnosis_of_discovery_and_validation==0]

# Exclude repeated subject


# Overlaped subjects with FC_1322
id_fc_1322_overlaped_with_validation = pd.merge(id_fc_1322, id_validation, left_on=0, right_on=0, how='inner')
id_fc_1322_overlaped_with_repeated_subjects_with_discovery = pd.merge(id_fc_1322, id_repeat_with_discovery, left_on=0, right_on=0, how='inner')

# Save
id_repeated_discovery = pd.merge(id_fc_1322, repeated_paire_in_discovery, left_on=0, right_on=1, how='inner')
repeated_pair_of_discovery_selected = pd.merge(id_fc_1322_overlaped_with_repeated_subjects_with_discovery, repeatPairs_only_twice, left_on=0, right_on=1, how='inner').iloc[:,[2,0]]
repeated_pair_of_discovery_selected.columns = ['first id','second id']

id_fc_1322_overlaped_with_repeated_subjects_with_discovery.to_excel(r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\id_repeated_with_discovery.xlsx', header=None, index=None)
repeated_pair_of_discovery_selected.to_excel(r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\id_repeated_pair_of_discovery_selected.xlsx', header=None, index=None)

# head motion
headmotion = pd.read_excel(headmotion_file)
headmotion_repeat_with_discovery = pd.merge(id_repeat_with_discovery, headmotion, left_on=0, right_on='Subject ID', how='inner')['mean FD_Power']
headmotion_repeat_with_discovery.describe()

#%% Identify changed diagnosis for all repeated subjects
diagnosis_first = pd.merge(repeatPairs_only_twice[0],scale, left_on=0, right_on='folder', how='left')[['folder', '诊断']]
diagnosis_second = pd.merge( repeatPairs_only_twice[1], scale, left_on=1, right_on='folder', how='left')[['folder', '诊断']]
diagnosis_changed_loc = (diagnosis_first['诊断'] - diagnosis_second['诊断']) != 0
changed_before = diagnosis_first.iloc[diagnosis_changed_loc.values, :]
changed_after = diagnosis_second.iloc[diagnosis_changed_loc.values, :]

