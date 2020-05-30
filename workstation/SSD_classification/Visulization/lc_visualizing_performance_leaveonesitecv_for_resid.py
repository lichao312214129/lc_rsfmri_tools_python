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
classification_results_all = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_fc_excluded_greater_fd_and_regressed_out_site_sex_motion_all.npy'
classification_results_separately = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_fc_excluded_greater_fd_and_regressed_out_sex_motion_separately.npy'

is_plot = 1
is_savefig = 1

#%% Load and proprocess
scale_550 = pd.read_excel(scale_550_file)
scale_550['folder'] = np.int32(scale_550['folder'])

results_all_leave_one_site_cv = np.load(classification_results_all, allow_pickle=True)
results_special_all = results_all_leave_one_site_cv['special_result']
results_special_all = pd.DataFrame(results_special_all)
results_special_all.iloc[:, 0] = np.int32(results_special_all.iloc[:, 0])

# Filter subjects that have .mat files
scale_550_selected_all = pd.merge(results_special_all, scale_550, left_on=0, right_on='folder', how='inner')


#%% Calculate performance for Schizophrenia Spectrum subgroups
duration = 18  # Upper limit of first episode: 
# Frist episode unmedicated; first episode medicated; chronic medicated
# all_svc
data_chronic_medicated_SSD_550_18_all = scale_550_selected_all[
    (scale_550_selected_all['诊断']==3) & 
    (scale_550_selected_all['病程月'] > duration) & 
    (scale_550_selected_all['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_all = scale_550_selected_all[
    (scale_550_selected_all['诊断']==3) & 
    (scale_550_selected_all['首发'] == 1) &
    (scale_550_selected_all['病程月'] <= duration) & 
    (scale_550_selected_all['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_all = scale_550_selected_all[
    (scale_550_selected_all['诊断']==3) & 
    (scale_550_selected_all['首发'] == 1) &
    (scale_550_selected_all['病程月'] <= duration) & 
    (scale_550_selected_all['用药'] == 0)
]

#%% Calculating Accuracy
# all
acc_chronic_medicated_SSD_550_18_all = np.sum(data_chronic_medicated_SSD_550_18_all[1]-data_chronic_medicated_SSD_550_18_all[3]==0) / len(data_chronic_medicated_SSD_550_18_all)
acc_firstepisode_medicated_SSD_550_18_all = np.sum(data_firstepisode_medicated_SSD_550_18_all[1]-data_firstepisode_medicated_SSD_550_18_all[3]==0) / len(data_firstepisode_medicated_SSD_550_18_all)
acc_first_episode_unmedicated_SSD_550_18_all = np.sum(data_firstepisode_unmedicated_SSD_550_18_all[1]-data_firstepisode_unmedicated_SSD_550_18_all[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_all)
accuracy_all, sensitivity_all, specificity_all, auc_all = eval_performance(scale_550_selected_all[1].values, scale_550_selected_all[3].values, scale_550_selected_all[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

#%% Statistics
# all
n = len(data_chronic_medicated_SSD_550_18_all)
acc = acc_chronic_medicated_SSD_550_18_all
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_all)
acc = acc_firstepisode_medicated_SSD_550_18_all
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_all)
acc = acc_first_episode_unmedicated_SSD_550_18_all
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

#%% ---------------------------------------------------------------------------------------------------------
results_leave_one_site_cv = np.load(classification_results_separately, allow_pickle=True)
results_special = results_leave_one_site_cv['special_result']
results_special = pd.DataFrame(results_special)
results_special.iloc[:, 0] = np.int32(results_special.iloc[:, 0])

# Filter subjects that have .mat files
scale_550_selected = pd.merge(results_special, scale_550, left_on=0, right_on='folder', how='inner')


#%% Calculate performance for Schizophrenia Spectrum subgroups
duration = 18  # Upper limit of first episode: 
# Frist episode unmedicated; first episode medicated; chronic medicated
# all_svc
data_chronic_medicated_SSD_550_18 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['病程月'] > duration) & 
    (scale_550_selected['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['首发'] == 1) &
    (scale_550_selected['病程月'] <= duration) & 
    (scale_550_selected['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18 = scale_550_selected[
    (scale_550_selected['诊断']==3) & 
    (scale_550_selected['首发'] == 1) &
    (scale_550_selected['病程月'] <= duration) & 
    (scale_550_selected['用药'] == 0)
]

#%% Calculating Accuracy
# all
acc_chronic_medicated_SSD_550_18 = np.sum(data_chronic_medicated_SSD_550_18[1]-data_chronic_medicated_SSD_550_18[3]==0) / len(data_chronic_medicated_SSD_550_18)
acc_firstepisode_medicated_SSD_550_18 = np.sum(data_firstepisode_medicated_SSD_550_18[1]-data_firstepisode_medicated_SSD_550_18[3]==0) / len(data_firstepisode_medicated_SSD_550_18)
acc_first_episode_unmedicated_SSD_550_18 = np.sum(data_firstepisode_unmedicated_SSD_550_18[1]-data_firstepisode_unmedicated_SSD_550_18[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18)
accuracy, sensitivity, specificity, auc = eval_performance(scale_550_selected[1].values, scale_550_selected[3].values, scale_550_selected[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

#%% Statistics
# all
n = len(data_chronic_medicated_SSD_550_18)
acc = acc_chronic_medicated_SSD_550_18
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18)
acc = acc_firstepisode_medicated_SSD_550_18
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18)
acc = acc_first_episode_unmedicated_SSD_550_18
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

#%% Plot
plt.figure(figsize=(5,8))
ax1=plt.bar(
    [0, 2.5, 5, 7.5, 10, 12.5, ], [
        accuracy_all, 
        sensitivity_all, 
        specificity_all, 
        acc_chronic_medicated_SSD_550_18_all, 
        acc_firstepisode_medicated_SSD_550_18_all, 
        acc_first_episode_unmedicated_SSD_550_18_all,
    ], 
    # color=['blue'],
    alpha=0.4)

ax2 = plt.bar(
    [1, 3.5, 6, 8.5, 11, 13.5 ], 
    [
        accuracy, 
        sensitivity, 
        specificity,
        acc_chronic_medicated_SSD_550_18,
        acc_firstepisode_medicated_SSD_550_18, 
        acc_first_episode_unmedicated_SSD_550_18
    ], 
    color=['green'],
    alpha=0.4)
plt.grid(axis='x')

plt.yticks(fontsize=12)
plt.xticks([0.5, 3, 5.5, 8, 10.5, 13], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.legend([ax1, ax2], ['Correction pooled all datasets', 'Correction in training dataset'], loc='upper right')

plt.subplots_adjust(wspace = 0.5, hspace =1)
plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\Correaction of site and cov.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)


# ax1=plt.bar(
#     [0,1,2,3,4,5], [
#         accuracy_all, 
#         sensitivity_all, 
#         specificity_all, 
#         acc_chronic_medicated_SSD_550_18_all, 
#         acc_firstepisode_medicated_SSD_550_18_all, 
#         acc_first_episode_unmedicated_SSD_550_18_all,
#     ], 
#     # color=['b'],
#     alpha=0.4)
# plt.yticks(fontsize=12)
# plt.grid(axis='x')
# plt.xticks([0,1,2,3,4,5],['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
# # plt.legend([ax1, ax2], ['Regressed out site, sex and head motion', 'Without regression'], loc='upper right')
# plt.tight_layout()
# plt.show()


