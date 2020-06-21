# -*- coding: utf-8 -*-
""" Move healthy controls to another folder for clustering using HYDRA.

@author: Li Chao
"""

import os
import pandas as pd

import eslearn.utils.lc_copy_selected_file_V6 as lcopy

#%% Inputs
scale_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\10-24大表.xlsx'
id_discovery_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\id.xlsx'
out_path = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale'

# Load
scale = pd.read_excel(scale_file)
id_discovery = pd.read_excel(id_discovery_file, header=None)

# Extract HCs id
id_hc = pd.merge(id_discovery, scale, left_on=0, right_on='folder', how='inner')
id_hc = id_hc['folder'][id_hc['诊断']==1]

# Save HCs id
id_hc.to_excel(os.path.join(out_path, "id_hc.xlsx"), header=None, index=None)

# Move
uid =  r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\id_hc.xlsx'
source_folder = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\zFC_610'
out_path = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\zFC_HC_210'

unique_id_level_of_target_file = 1
keywork_of_target_file = ''
save_suffix= ''

copy = lcopy.CopyFmri(
        reference_file=uid,
        targe_file_folder=source_folder,
        keywork_of_reference_uid='([1-9]\d*)',
        ith_number_of_reference_uid=0,
        keyword_of_target_file_uid='([1-9]\d*)',
        ith_number_of_targetfile_uid=0,
        unique_id_level_of_target_file=unique_id_level_of_target_file,
        keywork_of_target_file=keywork_of_target_file,
        keyword_of_parent_folder_containing_target_file='',
        out_path=out_path,
        n_processess=8,
        is_save_log=0,
        is_copy=0,
        is_move=1,
        save_into_one_or_more_folder='all_files_in_one_folder',
        save_suffix=save_suffix,
        is_run=1)

results = copy.main_run()
