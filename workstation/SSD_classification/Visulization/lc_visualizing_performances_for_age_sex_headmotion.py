# -*- coding: utf-8 -*-
"""This script is used to visualize age, sex, headmoion and correlation between age and duration
for each group of each site
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
from eslearn.visualization.el_violine import ViolinPlotMatplotlib, ViolinPlot
from eslearn.utils.lc_evaluation_model_performances import eval_performance
from eslearn.utils.lc_read_write_mat import read_mat, write_mat

#%% Inputs
age_p_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\age_p_site1.mat'
age_c_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\age_c_site1.mat'
age_p_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\age_p_site234.mat'
age_c_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\age_c_site234.mat'

sex_p_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\sex_p_site1.mat'
sex_c_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\sex_c_site1.mat'
sex_p_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\sex_p_site234.mat'
sex_c_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\sex_c_site234.mat'

headmotion_p_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site1.mat'
headmotion_c_site1_file = r'D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site1.mat'
headmotion_p_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site234.mat'
headmotion_c_site234_file = r'D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site234.mat'

is_plot = 1
is_savefig = 1

#%% Load
age_p_site1 = read_mat(age_p_site1_file)
age_c_site1 = read_mat(age_c_site1_file)
age_p_site234 = read_mat(age_p_site234_file)
age_c_site234 = read_mat(age_c_site234_file)

sex_p_site1 = read_mat(sex_p_site1_file)
sex_c_site1 = read_mat(sex_c_site1_file)
sex_p_site234 = read_mat(sex_p_site234_file)
sex_c_site234 = read_mat(sex_c_site234_file)

headmotion_p_site1 = read_mat(headmotion_p_site1_file)
headmotion_c_site1 = read_mat(headmotion_c_site1_file)
headmotion_p_site234 = read_mat(headmotion_p_site234_file)
headmotion_c_site234 = read_mat(headmotion_c_site234_file)

#%% Plot
plt.figure(figsize=(10,6))

# Age
plt.subplot(1,3,1)
ViolinPlotMatplotlib().plot([age_p_site234, age_c_site234, age_p_site1, age_c_site1], 
                            facecolor=['red', 'green', 'red', 'green'], positions=[0,1, 2.5,3.5])

plt.xticks([ 0.5, 3], ['Training datasets', 'Test dataset'], rotation=45, ha="right")  
plt.legend(['SSD', 'HC'])
plt.ylabel('Year',fontsize=12) 
plt.title('Age', fontweight='bold', fontsize=15)

# Sex
plt.subplot(1,3,2)
ax = plt.bar(
    [0,0.5,1.5,2],
    [
        (sum(sex_p_site234==1)/len(sex_p_site234))[0], 
        (sum(sex_c_site234==1)/len(sex_c_site234))[0], 
        (sum(sex_p_site1==1)/len(sex_p_site1))[0],
        (sum(sex_c_site1==1)/len(sex_c_site1))[0]
    ], 
    color=['red', 'green','red', 'green'], 
    width=0.4, 
    alpha=0.35
)

plt.xticks([ 0.25, 1.75], ['Training datasets', 'Test dataset'], rotation=45, ha="right")  
plt.legend([ax[0], ax[1]], ['SSD', 'HC'])
plt.ylabel('Proportion of male',fontsize=12) 
plt.title('Gender', fontweight='bold', fontsize=15)


# Head motion
plt.subplot(1,3,3)
ViolinPlotMatplotlib().plot([headmotion_p_site234, headmotion_c_site234, headmotion_p_site1, headmotion_c_site1], 
                            facecolor=['red', 'green', 'red', 'green'], positions=[0,1, 2.5,3.5])

plt.xticks([ 0.5, 3], ['Training datasets', 'Test dataset'], rotation=45, ha="right")  
plt.legend(['SSD', 'HC'])
plt.ylabel('Mean FD',fontsize=12) 
plt.title('Head motion', fontweight='bold', fontsize=15)

plt.subplots_adjust(wspace = 0.2, hspace =0.2)
plt.tight_layout()
# pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\age_sex_headmotion.pdf')
# pdf.savefig()
# pdf.close()
plt.show()
print('-'*50)
