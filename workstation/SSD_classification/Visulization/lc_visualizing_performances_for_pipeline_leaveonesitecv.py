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
classification_results_pca70_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pca70_svc_leave_one_site_cv.npy'
classification_results_pca80_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pca80_svc_leave_one_site_cv.npy'
classification_results_pca99_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pca99_svc_leave_one_site_cv.npy'
classification_results_pca95_lr_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pca95_lr_leave_one_site_cv.npy'
is_plot = 1
is_savefig = 1

#%% Load and proprocess
scale_550 = pd.read_excel(scale_550_file)
scale_550['folder'] = np.int32(scale_550['folder'])

results_pca70_leave_one_site_cv = np.load(classification_results_pca70_leave_one_site_cv_file, allow_pickle=True)
results_special_pca70 = results_pca70_leave_one_site_cv['special_result']
results_special_pca70 = pd.DataFrame(results_special_pca70)
results_special_pca70.iloc[:, 0] = np.int32(results_special_pca70.iloc[:, 0])

results_pca80_leave_one_site_cv = np.load(classification_results_pca80_leave_one_site_cv_file, allow_pickle=True)
results_special_pca80 = results_pca80_leave_one_site_cv['special_result']
results_special_pca80 = pd.DataFrame(results_special_pca80)
results_special_pca80.iloc[:, 0] = np.int32(results_special_pca80.iloc[:, 0])

results_pca99_leave_one_site_cv = np.load(classification_results_pca99_leave_one_site_cv_file, allow_pickle=True)
results_special_pca99 = results_pca99_leave_one_site_cv['special_result']
results_special_pca99 = pd.DataFrame(results_special_pca99)
results_special_pca99.iloc[:, 0] = np.int32(results_special_pca99.iloc[:, 0])

results_pca95_lr_leave_one_site_cv = np.load(classification_results_pca95_lr_leave_one_site_cv_file, allow_pickle=True)
results_special_pca95_lr = results_pca95_lr_leave_one_site_cv['special_result']
results_special_pca95_lr = pd.DataFrame(results_special_pca95_lr)
results_special_pca95_lr.iloc[:, 0] = np.int32(results_special_pca95_lr.iloc[:, 0])

# Filter subjects that have .mat files
scale_550_selected_pca70 = pd.merge(results_special_pca70, scale_550, left_on=0, right_on='folder', how='inner')
scale_550_selected_pca80 = pd.merge(results_special_pca80, scale_550, left_on=0, right_on='folder', how='inner')
scale_550_selected_pca99 = pd.merge(results_special_pca99, scale_550, left_on=0, right_on='folder', how='inner')
scale_550_selected_pca95_lr = pd.merge(results_special_pca95_lr, scale_550, left_on=0, right_on='folder', how='inner')

#%% Calculate performance for Schizophrenia Spectrum subgroups
duration = 18  # Upper limit of first episode: 
# Frist episode unmedicated; first episode medicated; chronic medicated
# pca70_svc
data_chronic_medicated_SSD_550_18_pca70 = scale_550_selected_pca70[
    (scale_550_selected_pca70['诊断']==3) & 
    (scale_550_selected_pca70['病程月'] > duration) & 
    (scale_550_selected_pca70['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_pca70 = scale_550_selected_pca70[
    (scale_550_selected_pca70['诊断']==3) & 
    (scale_550_selected_pca70['首发'] == 1) &
    (scale_550_selected_pca70['病程月'] <= duration) & 
    (scale_550_selected_pca70['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_pca70 = scale_550_selected_pca70[
    (scale_550_selected_pca70['诊断']==3) & 
    (scale_550_selected_pca70['首发'] == 1) &
    (scale_550_selected_pca70['病程月'] <= duration) & 
    (scale_550_selected_pca70['用药'] == 0)
]

# pca80_svc
data_chronic_medicated_SSD_550_18_pca80 = scale_550_selected_pca80[
    (scale_550_selected_pca80['诊断']==3) & 
    (scale_550_selected_pca80['病程月'] > duration) & 
    (scale_550_selected_pca80['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_pca80 = scale_550_selected_pca80[
    (scale_550_selected_pca80['诊断']==3) & 
    (scale_550_selected_pca80['首发'] == 1) &
    (scale_550_selected_pca80['病程月'] <= duration) & 
    (scale_550_selected_pca80['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_pca80 = scale_550_selected_pca80[
    (scale_550_selected_pca80['诊断']==3) & 
    (scale_550_selected_pca80['首发'] == 1) &
    (scale_550_selected_pca80['病程月'] <= duration) & 
    (scale_550_selected_pca80['用药'] == 0)
]

# pca99_svc
data_chronic_medicated_SSD_550_18_pca99 = scale_550_selected_pca99[
    (scale_550_selected_pca99['诊断']==3) & 
    (scale_550_selected_pca99['病程月'] > duration) & 
    (scale_550_selected_pca99['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_pca99 = scale_550_selected_pca99[
    (scale_550_selected_pca99['诊断']==3) & 
    (scale_550_selected_pca99['首发'] == 1) &
    (scale_550_selected_pca99['病程月'] <= duration) & 
    (scale_550_selected_pca99['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_pca99 = scale_550_selected_pca99[
    (scale_550_selected_pca99['诊断']==3) & 
    (scale_550_selected_pca99['首发'] == 1) &
    (scale_550_selected_pca99['病程月'] <= duration) & 
    (scale_550_selected_pca99['用药'] == 0)
]

# pca95_lr_svc
data_chronic_medicated_SSD_550_18_pca95_lr = scale_550_selected_pca95_lr[
    (scale_550_selected_pca95_lr['诊断']==3) & 
    (scale_550_selected_pca95_lr['病程月'] > duration) & 
    (scale_550_selected_pca95_lr['用药'] == 1)
]
data_firstepisode_medicated_SSD_550_18_pca95_lr = scale_550_selected_pca95_lr[
    (scale_550_selected_pca95_lr['诊断']==3) & 
    (scale_550_selected_pca95_lr['首发'] == 1) &
    (scale_550_selected_pca95_lr['病程月'] <= duration) & 
    (scale_550_selected_pca95_lr['用药'] == 1)
]
data_firstepisode_unmedicated_SSD_550_18_pca95_lr = scale_550_selected_pca95_lr[
    (scale_550_selected_pca95_lr['诊断']==3) & 
    (scale_550_selected_pca95_lr['首发'] == 1) &
    (scale_550_selected_pca95_lr['病程月'] <= duration) & 
    (scale_550_selected_pca95_lr['用药'] == 0)
]

#%% Calculating Accuracy
# pca70
acc_chronic_medicated_SSD_550_18_pca70 = np.sum(data_chronic_medicated_SSD_550_18_pca70[1]-data_chronic_medicated_SSD_550_18_pca70[3]==0) / len(data_chronic_medicated_SSD_550_18_pca70)
acc_firstepisode_medicated_SSD_550_18_pca70 = np.sum(data_firstepisode_medicated_SSD_550_18_pca70[1]-data_firstepisode_medicated_SSD_550_18_pca70[3]==0) / len(data_firstepisode_medicated_SSD_550_18_pca70)
acc_first_episode_unmedicated_SSD_550_18_pca70 = np.sum(data_firstepisode_unmedicated_SSD_550_18_pca70[1]-data_firstepisode_unmedicated_SSD_550_18_pca70[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_pca70)
accuracy_pca70, sensitivity_pca70, specificity_pca70, auc_pca70 = eval_performance(scale_550_selected_pca70[1].values, scale_550_selected_pca70[3].values, scale_550_selected_pca70[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

# pca80
acc_chronic_medicated_SSD_550_18_pca80 = np.sum(data_chronic_medicated_SSD_550_18_pca80[1]-data_chronic_medicated_SSD_550_18_pca80[3]==0) / len(data_chronic_medicated_SSD_550_18_pca80)
acc_firstepisode_medicated_SSD_550_18_pca80 = np.sum(data_firstepisode_medicated_SSD_550_18_pca80[1]-data_firstepisode_medicated_SSD_550_18_pca80[3]==0) / len(data_firstepisode_medicated_SSD_550_18_pca80)
acc_first_episode_unmedicated_SSD_550_18_pca80 = np.sum(data_firstepisode_unmedicated_SSD_550_18_pca80[1]-data_firstepisode_unmedicated_SSD_550_18_pca80[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_pca80)
accuracy_pca80, sensitivity_pca80, specificity_pca80, auc_pca80 = eval_performance(scale_550_selected_pca80[1].values, scale_550_selected_pca80[3].values, scale_550_selected_pca80[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

# pca99
acc_chronic_medicated_SSD_550_18_pca99 = np.sum(data_chronic_medicated_SSD_550_18_pca99[1]-data_chronic_medicated_SSD_550_18_pca99[3]==0) / len(data_chronic_medicated_SSD_550_18_pca99)
acc_firstepisode_medicated_SSD_550_18_pca99 = np.sum(data_firstepisode_medicated_SSD_550_18_pca99[1]-data_firstepisode_medicated_SSD_550_18_pca99[3]==0) / len(data_firstepisode_medicated_SSD_550_18_pca99)
acc_first_episode_unmedicated_SSD_550_18_pca99 = np.sum(data_firstepisode_unmedicated_SSD_550_18_pca99[1]-data_firstepisode_unmedicated_SSD_550_18_pca99[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_pca99)
accuracy_pca99, sensitivity_pca99, specificity_pca99, auc_pca99 = eval_performance(scale_550_selected_pca99[1].values, scale_550_selected_pca99[3].values, scale_550_selected_pca99[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

# pca95_lr
acc_chronic_medicated_SSD_550_18_pca95_lr = np.sum(data_chronic_medicated_SSD_550_18_pca95_lr[1]-data_chronic_medicated_SSD_550_18_pca95_lr[3]==0) / len(data_chronic_medicated_SSD_550_18_pca95_lr)
acc_firstepisode_medicated_SSD_550_18_pca95_lr = np.sum(data_firstepisode_medicated_SSD_550_18_pca95_lr[1]-data_firstepisode_medicated_SSD_550_18_pca95_lr[3]==0) / len(data_firstepisode_medicated_SSD_550_18_pca95_lr)
acc_first_episode_unmedicated_SSD_550_18_pca95_lr = np.sum(data_firstepisode_unmedicated_SSD_550_18_pca95_lr[1]-data_firstepisode_unmedicated_SSD_550_18_pca95_lr[3]==0) / len(data_firstepisode_unmedicated_SSD_550_18_pca95_lr)
accuracy_pca95_lr, sensitivity_pca95_lr, specificity_pca95_lr, auc_pca95_lr = eval_performance(scale_550_selected_pca95_lr[1].values, scale_550_selected_pca95_lr[3].values, scale_550_selected_pca95_lr[2].values,
                    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                    verbose=True, is_showfig=False, legend1='HC', legend2='Patients', is_savefig=False, out_name=None)

#%% Statistics
# pca70
n = len(data_chronic_medicated_SSD_550_18_pca70)
acc = acc_chronic_medicated_SSD_550_18_pca70
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_pca70)
acc = acc_firstepisode_medicated_SSD_550_18_pca70
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_pca70)
acc = acc_first_episode_unmedicated_SSD_550_18_pca70
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

# pca80
n = len(data_chronic_medicated_SSD_550_18_pca80)
acc = acc_chronic_medicated_SSD_550_18_pca80
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_pca80)
acc = acc_firstepisode_medicated_SSD_550_18_pca80
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_pca80)
acc = acc_first_episode_unmedicated_SSD_550_18_pca80
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

# pca99
n = len(data_chronic_medicated_SSD_550_18_pca99)
acc = acc_chronic_medicated_SSD_550_18_pca99
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_pca99)
acc = acc_firstepisode_medicated_SSD_550_18_pca99
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_pca99)
acc = acc_first_episode_unmedicated_SSD_550_18_pca99
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

# pca_lc
n = len(data_chronic_medicated_SSD_550_18_pca95_lr)
acc = acc_chronic_medicated_SSD_550_18_pca95_lr
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_medicated_SSD_550_18_pca95_lr)
acc = acc_firstepisode_medicated_SSD_550_18_pca95_lr
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)
n = len(data_firstepisode_unmedicated_SSD_550_18_pca95_lr)
acc = acc_first_episode_unmedicated_SSD_550_18_pca95_lr
k = np.int32(n * acc)
p, sum_prob, prob, randk = lc_binomialtest(n, k, 0.5, 0.5)
print(p)

#%% Plot
plt.figure(figsize=(10,15))

plt.subplot(221)
plt.bar([0,1,2,3,4,5],
    [accuracy_pca70, sensitivity_pca70, specificity_pca70,
    acc_chronic_medicated_SSD_550_18_pca70, 
    acc_firstepisode_medicated_SSD_550_18_pca70, 
    acc_first_episode_unmedicated_SSD_550_18_pca70], 
    alpha=0.5
)
plt.yticks(fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.grid(axis='y')
plt.title('PCA_70_svc', fontsize=15, fontweight='bold')

plt.subplot(222)
plt.bar([0,1,2,3,4,5],
    [accuracy_pca80, sensitivity_pca80, specificity_pca80,
    acc_chronic_medicated_SSD_550_18_pca80, 
    acc_firstepisode_medicated_SSD_550_18_pca80, 
    acc_first_episode_unmedicated_SSD_550_18_pca80], 
    alpha=0.5
)
plt.yticks(fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.grid(axis='y')
plt.title('PCA_80_svc', fontsize=15, fontweight='bold')

plt.subplot(223)
plt.bar([0,1,2,3,4,5],
    [accuracy_pca99, sensitivity_pca99, specificity_pca99,
    acc_chronic_medicated_SSD_550_18_pca99, 
    acc_firstepisode_medicated_SSD_550_18_pca99, 
    acc_first_episode_unmedicated_SSD_550_18_pca99], 
    alpha=0.5
)
plt.yticks(fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.grid(axis='y')
plt.title('PCA_99_svc', fontsize=15, fontweight='bold')

plt.subplot(224)
plt.bar([0,1,2,3,4,5],
    [accuracy_pca95_lr, sensitivity_pca95_lr, specificity_pca95_lr,
    acc_chronic_medicated_SSD_550_18_pca95_lr, 
    acc_firstepisode_medicated_SSD_550_18_pca95_lr, 
    acc_first_episode_unmedicated_SSD_550_18_pca95_lr], 
    alpha=0.5
)
plt.yticks(fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['Accuracy', 'Sensitivity','Specificity', 'Sensitivity of chronic SSD', 'Sensitivity of first episode medicated SSD', 'Sensitivity of first episode unmedicated SSD'], rotation=45, ha="right")  
plt.grid(axis='y')
plt.title('PCA_95_lr', fontsize=15, fontweight='bold')

plt.subplots_adjust(wspace = 0.5, hspace =1)
plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\other_algorithm.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)


