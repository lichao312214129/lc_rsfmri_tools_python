# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:07:21 2019
用于给晓薇姐处理数据
@author: lenovo
"""
#
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\classfication')
from lc_read_nii import main
from lc_read_nii import read_sigleNii_LC
import numpy as np

# =====================================================================
patients_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\P_Weighted_selected'
hc_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected'
mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
# =====================================================================
is_train = 1
mask, _ = read_sigleNii_LC(mask)
mask = mask >= 0.2
mask = np.array(mask).reshape(-1,)

def load_nii_and_gen_label(patients_path, hc_path, mask):
    # data
    data1, _ = main(patients_path, '.img')
    data1 = np.squeeze(
        np.array([np.array(data1).reshape(1, -1) for data1 in data1]))

    data2, _ = main(hc_path, '.img')
    data2 = np.squeeze(
        np.array([np.array(data2).reshape(1, -1) for data2 in data2]))

    data = np.vstack([data1, data2])

    # data in mask
    data_in_mask = data[:, mask]
    # label
    label = np.hstack([np.ones([len(data1), ]) - 1, np.ones([len(data2), ])])
    return data, data_in_mask, label

# training and test


def tr_te():
    import lc_svc_rfe_cv_V2 as lsvc
    svc = lsvc.SVCRefCv(k=3,pca_n_component=0.9, show_results=1, show_roc=0)
    if is_train:
        results = svc.svc_rfe_cv(data_in_mask, label)
    return results


# run
data_2, data_in_mask, label = load_nii_and_gen_label(
    patients_path, hc_path, mask)
results = tr_te()
results = results.__dict__
