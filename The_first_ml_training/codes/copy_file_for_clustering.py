# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 10:03:02 2020

@author: Li Chao
Email: lichao19870617@163.com
"""

import shutil
import os
import scipy.io as sio  # scipy用来读取mat文件，scipy可以做统计

source = r"D:\My_Codes\lc_private_codes\The_first_ml_training\clustering\hydra\MDD"
target_root = r"D:\My_Codes\lc_private_codes\The_first_ml_training\clustering\hydra"

def copy_file(source, target_root, target_foldername="subtype1", file_name=None):
    file_path = [os.path.join(source, fn) for fn in file_name]
    target_path = [os.path.join(target_root, target_foldername, fn) for fn in file_name]
    
    # 判断有误目标文件夹
    if not os.path.exists(os.path.join(target_root, target_foldername)):
        os.makedirs(os.path.join(target_root, target_foldername))
        
    for src, tgt in zip(file_path, target_path):
        shutil.copy(src, tgt)

file_name1 = ['smReHoMap_033.nii', 'smReHoMap_036.nii']
copy_file(source, target_root, "subtype1", file_name1)



file_name2 = ['smReHoMap_028.nii', 'smReHoMap_029.nii', 'smReHoMap_035.nii']

#%% 全自动的把亚类分开
idx_file = r"D:\My_Codes\lc_private_codes\The_first_ml_training\clustering\hydra\subtype_index.mat"
idx = sio.loadmat(idx_file)

keys = list(idx.keys())
idx_data = idx[keys[-1]]

# 取出第一个类所有人的ID
for itype in range(4):
    id_subtype1 = idx_data[itype]
    n_subj = len(id_subtype1[0])  # 判断第一个亚类有几个人
    ids_subtype = [id_subtype1[0][i][0][0] for i in range(n_subj)]
    copy_file(source, target_root, f"subtype{itype+1}", ids_subtype)










