"""
This script is used to transform the 206 dataset into .npy format.
1.Transform the .mat files to one .npy file
2. Give labels to each subject, concatenate at the first column
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np 
import pandas as pd
import os
from Utils.lc_read_write_mat import read_mat

# Inputs
matroot = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\SelectedFC_206'  # all mat files directory
scale = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\SZ_NC_108_100.xlsx'  # whole scale path
n_node = 246  #  number of nodes in the mat network

# Transform the .mat files to one .npy file
allmatpath = os.listdir(matroot)
allmatpath = [os.path.join(matroot, matpath) for matpath in allmatpath]
mask = np.triu(np.ones(n_node),1)==1
allmat = [read_mat(matpath)[mask].T for matpath in allmatpath]
allmat = pd.DataFrame(np.float32(allmat))

# Give uid and labels to each subject, concatenate at the first column
uid = [os.path.basename(matpath) for matpath in allmatpath]
uid = pd.Series(uid)
uid = uid.str.findall('(NC.*[0-9]\d*|SZ.*[0-9]\d*)')
uid = [str(id[0]) for id in uid]
uid = pd.DataFrame([''.join(id.split('_')) for id in uid])

scale = pd.read_excel(scale)
selected_diagnosis = pd.merge(uid, scale, left_on=0, right_on='ID', how='inner')[['ID','group']]
selected_diagnosis['group'][selected_diagnosis['group']==2] = 0

allmat_plus_label = pd.concat([selected_diagnosis, allmat],axis=1)

allmat_plus_label['ID'] = allmat_plus_label['ID'].str.replace('NC','1');
allmat_plus_label['ID'] = allmat_plus_label['ID'].str.replace('SZ','2');
allmat_plus_label = pd.DataFrame(allmat_plus_label, dtype=np.float64)
np.save(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_206.npy',allmat_plus_label)
d= np.load(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_206.npy',allow_pickle=True)
print(d)
