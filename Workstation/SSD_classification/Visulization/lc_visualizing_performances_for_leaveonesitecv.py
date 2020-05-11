# -*- coding: utf-8 -*-
"""This script is used to get each subgroups' demographic information and visualization
"""

#%%
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
sys.path.append(r'D:\My_Codes\easylearn-fmri\eslearn\statistical_analysis')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import pickle
import seaborn as sns

from lc_binomialtest import lc_binomialtest
from eslearn.statistical_analysis.lc_anova import oneway_anova
from eslearn.statistical_analysis.lc_chisqure import lc_chisqure
from eslearn.statistical_analysis.lc_ttest2 import ttest2
from eslearn.visualization.el_violine import ViolinPlotMatplotlib
from eslearn.utils.lc_evaluation_model_performances import eval_performance

#%% Inputs
scale_550_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
headmotion_file_dataset1 = r'D:\WorkStation_2018\SZ_classification\Scale\头动参数_1322.xlsx'
classification_results_results_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_leave_one_site_cv.npy'
is_plot = 1
is_savefig = 1

#%% Load and proprocess
scale_550 = pd.read_excel(scale_550_file)
headmotion_dataset1 = pd.read_excel(headmotion_file_dataset1)[['Subject ID', 'mean FD_Power']]
scale_550 = pd.merge(scale_550, headmotion_dataset1, left_on='folder', right_on='Subject ID', how='inner')

results_leave_one_site_cv = np.load(classification_results_results_leave_one_site_cv_file, allow_pickle=True)
results_special = results_leave_one_site_cv['special_result']
results_special = pd.DataFrame(results_special)
results_special.iloc[:, 0] = np.int32(results_special.iloc[:, 0])

scale_550['folder'] = np.int32(scale_550['folder'])

# Filter subjects that have .mat files
scale_550_selected = pd.merge(results_special, scale_550, left_on=0, right_on='folder', how='inner')

#%% Calculate performance for Schizophrenia Spectrum subgroups
## Step 1: Dataset1
duration = 18  # Upper limit of first episode: 
""" reference:
1. Kane JM, Robinson DG, Schooler NR, et al. Comprehensive versus usual
community care for first-episode psychosis: 2-year outcomes from the NIMH
RAISE early treatment program. Am J Psychiatry. 2016;173(4):362-372. doi:10.1176/appi.ajp.2015.15050632.
2. Cognitive Impairment in Never-Medicated Individuals on the Schizophrenia Spectrum. doi:10.1001/jamapsychiatry.2020.0001"
"""

# Frist episode unmedicated; first episode medicated; chronic medicated
data_chronic_medicated_SSD_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['病程月'] > duration) & 
    (scale_550_selected['用药'] == 1)
]

data_firstepisode_medicated_SSD_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['首发'] == 1) &
    (scale_550_selected['病程月'] <= duration) & 
    (scale_550_selected['用药'] == 1)
]

data_firstepisode_unmedicated_SSD_550 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['首发'] == 1) &
    (scale_550_selected['病程月'] <= duration) & 
    (scale_550_selected['用药'] == 0)
]

## Calculating Accuracy
acc_chronic_medicated_SSD_550 = np.sum(data_chronic_medicated_SSD_550[1]-data_chronic_medicated_SSD_550[3]==0) / len(data_chronic_medicated_SSD_550)
acc_first_episode_unmedicated_SSD_550 = np.sum(data_firstepisode_unmedicated_SSD_550[1]-data_firstepisode_unmedicated_SSD_550[3]==0) / len(data_firstepisode_unmedicated_SSD_550)
acc_firstepisode_medicated_SSD_550 = np.sum(data_firstepisode_medicated_SSD_550[1]-data_firstepisode_medicated_SSD_550[3]==0) / len(data_firstepisode_medicated_SSD_550)
acc_all_SSD_550 = np.sum(scale_550_selected[1]-scale_550_selected[3]==0) / len(scale_550_selected)

eval_performance(scale_550_selected[1].values, scale_550_selected[3].values, scale_550_selected[2].values)
print('-'*50)

# Extract subjects' demographic data
subinfo_chronic_medicated = data_chronic_medicated_SSD_550[['folder', '年龄', '性别', '学历（年）', 'BPRS_Total', 'mean FD_Power', '病程月']]
subinfo_firstepisode_medicated = data_firstepisode_medicated_SSD_550[['folder', '年龄', '性别', '学历（年）', 'BPRS_Total', 'mean FD_Power', '病程月']]
subinfo_firstepisode_unmedicated = data_firstepisode_unmedicated_SSD_550[['folder', '年龄', '性别', '学历（年）', 'BPRS_Total', 'mean FD_Power', '病程月']]

## Save index and subjects' demographic data
subinfo_chronic_medicated[['folder']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\index_chronic.txt', index=False, header=False)
subinfo_firstepisode_medicated[['folder']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\index_firstepisode_medicated.txt', index=False, header=False)
subinfo_firstepisode_unmedicated[['folder']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\index_firstepisode_unmedicated.txt', index=False, header=False)

subinfo_chronic_medicated[['folder', '年龄', '性别', '学历（年）', 'mean FD_Power', '病程月']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_chronic.txt', index=False)
subinfo_firstepisode_medicated[['folder', '年龄', '性别', '学历（年）', 'mean FD_Power', '病程月']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_firstepisode_medicated.txt', index=False)
subinfo_firstepisode_unmedicated[['folder', '年龄', '性别', '学历（年）', 'mean FD_Power', '病程月']].to_csv(r'D:\WorkStation_2018\SZ_classification\Scale\cov_firstepisode_unmedicated.txt', index=False)

plt.figure(figsize=(8,10))
title_dict = {0:'Age', 1:'Education', 2:'BPRS', 3:'Head motion', 4: 'Gender', 5:'Duration'}
ylabel_dict = {0:'Year', 1:'Year', 2:'', 3:'', 4:'Proportion of male', 5:'Month'}
for i, df in enumerate(['年龄', '学历（年）', 'BPRS_Total', 'mean FD_Power', '性别', '病程月']):
    plt.subplot(2,3,i+1)
    if i == 4:
        plt.bar(0, subinfo_chronic_medicated[df].dropna().value_counts()[1]/len(subinfo_chronic_medicated), width=0.3, alpha=0.5)
        plt.grid(axis='y')
        plt.bar(1, subinfo_firstepisode_medicated[df].dropna().value_counts()[1]/len(subinfo_firstepisode_medicated), width=0.3, alpha=0.5)
        plt.grid(axis='y')
        plt.bar(2, subinfo_firstepisode_unmedicated[df].dropna().value_counts()[1]/len(subinfo_firstepisode_unmedicated), width=0.3, alpha=0.5)
        plt.grid(axis='y')
        tt = [
            len(subinfo_chronic_medicated[df].dropna()),
            len(subinfo_firstepisode_medicated[df].dropna()),
            len(subinfo_firstepisode_unmedicated[df].dropna())
        ]
        obs = [
            subinfo_chronic_medicated[df].dropna().value_counts()[1],
            subinfo_firstepisode_medicated[df].dropna().value_counts()[1],
            subinfo_firstepisode_unmedicated[df].dropna().value_counts()[1]
        ]
        chi, p = lc_chisqure(obs, tt)
    else:
        ViolinPlotMatplotlib().plot([subinfo_chronic_medicated[df].dropna().values], positions=[0])
        plt.grid(axis='y')
        ViolinPlotMatplotlib().plot([subinfo_firstepisode_medicated[df].dropna().values], positions=[1])
        plt.grid(axis='y')
        ViolinPlotMatplotlib().plot([subinfo_firstepisode_unmedicated[df].dropna().values], positions=[2])
        plt.grid(axis='y')
        
        f, p = oneway_anova(
            *[subinfo_chronic_medicated[df].dropna(), 
            subinfo_firstepisode_medicated[df].dropna(), 
            subinfo_firstepisode_unmedicated[df].dropna()]
        )

    plt.yticks(fontsize=12)
    plt.xticks([0, 1, 2], ['Chronic SSD', 'First episode medicated SSD', 'First episode unmedicated SSD'], rotation=45, ha="right")
    plt.ylabel(ylabel_dict[i], fontsize=15)    
    plt.title(''.join([title_dict[i], f'(P={p:.2f})']), fontsize=12, fontweight="bold")

    
plt.subplots_adjust(wspace = 0.2, hspace =0)
plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\subgroupinfo.pdf')
pdf.savefig()
pdf.close()
plt.show()

# plt.figure(figsize=(10,5))
# sns.distplot(scale_550_selected[2][scale_550_selected[1]==1])
# sns.distplot(scale_550_selected[2][scale_550_selected[1]==0])
# ttest2(scale_550_selected[2][scale_550_selected[1]==1], scale_550_selected[2][scale_550_selected[1]==0])

