# -*- coding: utf-8 -*-
"""
This script is used to perform post-hoc analysis and visualization: 
the classification performance of subsets (only for Schizophrenia Spectrum: SZ and Schizophreniform).
Unless otherwise specified, all results  are for Schizophrenia Spectrum.
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

from lc_binomialtest import lc_binomialtest
from eslearn.statistical_analysis.lc_anova import oneway_anova
from eslearn.visualization.el_violine import ViolinPlot

#%% Inputs
classification_results_pooling_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_real_pooled.npy'
classification_results_results_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_fc_excluded_greater_fd_and_regressed_out_site_sex_motion_all.npy'
classification_results_feu_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_real_feu.npy'
is_plot = 1
is_savefig = 1

#%% Load and proprocess
results_pooling = np.load(classification_results_pooling_file, allow_pickle=True)
results_leave_one_site_cv = np.load(classification_results_results_leave_one_site_cv_file, allow_pickle=True)
results_feu = np.load(classification_results_feu_file, allow_pickle=True)

#%% -------------------------------------Visualization-----------------------------------------------
if is_plot:
    accuracy_pooling, sensitivity_pooling, specificity_pooling = results_pooling[0,:], results_pooling[1,:], results_pooling[2,:]
    performances_pooling = [accuracy_pooling, sensitivity_pooling, specificity_pooling]
    performances_pooling = pd.DataFrame(performances_pooling)

    # accuracy_leave_one_site_cv = results_leave_one_site_cv['accuracy']
    # sensitivity_leave_one_site_cv = results_leave_one_site_cv['sensitivity']
    # specificity_leave_one_site_cv = results_leave_one_site_cv['specificity']
    # performances_leave_one_site_cv = [accuracy_leave_one_site_cv, sensitivity_leave_one_site_cv, specificity_leave_one_site_cv]
    # performances_leave_one_site_cv = pd.DataFrame(performances_leave_one_site_cv)

    accuracy_feu, sensitivity_feu, specificity_feu = results_feu[0,:], results_feu[1,:], results_feu[2,:]
    performances_feu = [accuracy_feu, sensitivity_feu, specificity_feu]
    performances_feu = pd.DataFrame(performances_feu)
    
    # Bar: performances in the whole Dataset.
    import seaborn as sns
    plt.subplot(4,1,4)
    all_mean = np.concatenate([np.mean(performances_pooling.values,1), np.mean(performances_feu.values,1)])
    error = np.concatenate([np.std(performances_pooling.values, 1), np.std(performances_feu.values, 1)])

    color = ['darkturquoise'] * 3 +  ['paleturquoise'] * 3
    plt.bar(np.arange(0,len(all_mean)), all_mean, yerr = error, 
            capsize=5, linewidth=2, color=color)
    plt.tick_params(labelsize=10)
    plt.xticks(np.arange(0,len(all_mean)), ['Accuracy', 'Sensitivity', 'Sensitivity'] * 3, fontsize=10, rotation=45, ha='right')
    plt.title('Classification performances', fontsize=15, fontweight='bold')
    y_major_locator=MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(axis='y')
    plt.fill_between(np.linspace(-0.4,2.4), 1.01, 1.08, color='darkturquoise')
    plt.fill_between(np.linspace(2.6, 5.4), 1.01, 1.08, color='paleturquoise')

    # Save to PDF format
    if is_savefig & is_plot:
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.5, hspace = 0.5)  # wspace 左右
        pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Classification_performances_all_cutoff' + str(18) + '_resid.pdf')
        pdf.savefig()
        pdf.close()
    plt.show()
            