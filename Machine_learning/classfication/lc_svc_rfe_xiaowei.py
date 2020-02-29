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

# input begin

# 数据
patients_path=r'J:\Data_Code\晓薇姐\ALFF_MDD'
controls_path=r'J:\Data_Code\晓薇姐\ALFF_BD'

# mask
#mask=r'G:\Softer_DataProcessing\DPABI_V4.0_190305\Templates\WhiteMask_09_61x73x61.img'
mask=r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
mask=read_sigleNii_LC(mask)>=0.2
mask=np.array(mask).reshape(-1,)

if_training=1

# input end

def load_nii_and_gen_label(patients_path,controls_path,mask):
    # data
    data_p=main(patients_path)
    data_p=np.squeeze(np.array([np.array(data_p).reshape(1,-1) for data_p in data_p]))
    data_hc=main(controls_path)
    data_hc=np.squeeze(np.array([np.array(data_hc).reshape(1,-1) for data_hc in data_hc]))
    data=np.vstack([data_p,data_hc])
    # data in mask
    data_in_mask=data[:,mask]
    # label
    label=np.hstack([np.ones([len(data_p),]),np.ones([len(data_hc),])-2])
    return data,data_in_mask,label

# training and test
def tr_te():
    import lc_svc_rfe_cv_V2 as lsvc
    svc=lsvc.SVCRefCv(pca_n_component=0.85,show_results=1,show_roc=0,k=5)
    if if_training:
        results=svc.svc_rfe_cv(data_in_mask,label)
    return results

# run
data_2,data_in_mask,label=load_nii_and_gen_label(patients_path,controls_path,mask)
results=tr_te()
results=results.__dict__