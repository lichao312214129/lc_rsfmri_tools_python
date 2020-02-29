# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:07:21 2019
用于给晓薇姐处理数据
@author: lenovo
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Machine_learning\classfication')
from lc_read_nii import main
from lc_read_nii import read_sigleNii_LC
import numpy as np


# =============================================================================
# input begin
# 数据
BD_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\Weighted'
MDD_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'
HC_path = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\Weighted'
# =============================================================================

# mask
# mask=r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
# mask=read_sigleNii_LC(mask)>=0.2
# mask=np.array(mask).reshape(-1,)

if_training=1
# input end


def load_nii_and_gen_label(BD_path, MDD_path, HC_path, suffix = '.nii'):
    # data
    data1, _ = main(BD_path, suffix)
    data1 = np.squeeze(np.array([np.array(data1).reshape(1, -1) for data1 in data1]))
    
    data2, _ = main(MDD_path, suffix)
    data2 = np.squeeze(np.array([np.array(data2).reshape(1, -1) for data2 in data2]))
    
    data3, _ =main(HC_path, suffix)
    data3 = np.squeeze(np.array([np.array(data3).reshape(1, -1) for data3 in data3]))
    
    data=np.vstack([data1,data2,data3])
    
    # data in mask
    # data_in_mask=data[:,mask]
    # label
    label=np.hstack([np.ones([len(data1),])-1,np.ones([len(data2),]),np.ones([len(data2),])+1])
    return data, label

# training and test
def tr_te(data, label):
    import lc_svc_rfe_cv_V2 as lsvc
    svc=lsvc.SVCRefCv(pca_n_component=0.85,show_results=1,show_roc=0,k=2)
    if if_training:
        results=svc.svc_rfe_cv(data,label)
    return results

# run
data, label=load_nii_and_gen_label(BD_path,MDD_path, HC_path)
results=tr_te(data, label)
results=results.__dict__