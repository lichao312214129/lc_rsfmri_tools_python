# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:23:14 2018

@author: lenovo
"""
# =============================================================================
import sys
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\plot')
#import lc_boxplot as boxplot
import lc_violinplot as violin
import numpy as np
import pandas as pd


data_path=r'D:\myCodes\MVPA_LIChao\MVPA_Python\plot\scale_predct_label.xlsx'
#prd=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Machine_Learning\predictLabel_testData.xlsx'        
#prd_label=pd.read_excel(prd)
#a=pd.read_excel(data_path)
# =============================================================================
#增加
sel=violin.violinplot(
                    data_path=r'D:\myCodes\MVPA_LIChao\MVPA_Python\plot\scale_predct_label.xlsx',
                    x_location=[7,8],
                    x_name='脑区',
                    y_name='reho',
                    hue_name='predict_label',
                    hue_order=None,
                    if_save_figure=0,
                    figure_name=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\hamd_hama.tif')

r=sel.plot()
#
r.ax.set_xticklabels(labels=['HAMD','HAMA'],fontsize=20)
r.ax.set_ylabel(ylabel='',fontsize=20)
r.ax.set_xlabel(xlabel='',fontsize=20)

#sel.f.savefig(sel.figure_name, dpi=300, bbox_inches='tight')
