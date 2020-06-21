# -*- coding: utf-8 -*-
""" Compare demographics and clinical information among the subtypes

@author: Li Chao
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from eslearn.visualization.el_violine import ViolinPlotMatplotlib as vp

# subtype index
id_type = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\subtype_index.mat'
scale_file = r'D:\WorkStation_2018\Workstation_2020_transdiagnositic_subtyping\Data\Scale\大表_add_drugInfo.xlsx'

# Load
scale = pd.read_excel(scale_file)
type = sio.loadmat(id_type)

# Extract each subtype's index
type_data = type['subtype_index']
type_data = [pd.DataFrame(td[0]) for td in type_data]

# Extract each subtype's demographics and clinical information
header = [
    '年龄', '学历（年）', 'Age_of_first_episode', '病程月', 
    'HAMA_Total', 'HAMD-17_Total', 'YMRS_Total', 'BPRS_Total', 
    'Wisconsin_Card_Sorting_Test_CR,Correct_Responses', 'MCCB套别',
    '斯奈斯', '快乐体验能力量表总分', '期待性快感', '消费性快感',
    '性别', '首发', '用药_y', '自杀观念', '自杀行为',
    ]
info = [pd.merge(td, scale, left_on=0, right_on='folder', how='inner')[header] for td in type_data]
# info[2] = info[2].drop(0).dropna()  # The first subject of the subtype 2 missed most info

# Describe each subtype's demographics and clinical information
description = [info_[header].describe() for info_ in info]

# Visualization of  continuous variables
header = [
    '年龄', '学历（年）', 'Age_of_first_episode', '病程月', 
    'HAMA_Total', 'HAMD-17_Total', 'YMRS_Total', 'BPRS_Total', 
    'Wisconsin_Card_Sorting_Test_CR,Correct_Responses', 'MCCB套别',
    '斯奈斯', '快乐体验能力量表总分', '期待性快感', '消费性快感',
    ]
info_continuous = [info_[header].dropna(axis=1) for info_ in info]
info_continuous .pop(2)

plt.figure()
plt.subplot(2,4,1)
vp().plot([
            info[0]['年龄'].dropna().values, 
            info[1]['年龄'].dropna().values,
            info[2]['年龄'].dropna().values,
            info[3]['年龄'].dropna().values,
            info[4]['年龄'].dropna().values,
    ])
plt.title('Age')
plt.xticks(np.arange(1, 5), ['1', '2', '3', '4'])

plt.subplot(2,4,2)
vp().plot([
            info[0]['学历（年）'].dropna().values, 
            info[1]['学历（年）'].dropna().values,
            info[2]['学历（年）'].dropna().values,
            info[3]['学历（年）'].dropna().values,
            info[4]['学历（年）'].dropna().values,
    ])
plt.title('Education')

plt.subplot(2,4,3)
vp().plot([
            info[0]['HAMA_Total'].dropna().values, 
            info[1]['HAMA_Total'].dropna().values,
            info[2]['HAMA_Total'].dropna().values,
            info[3]['HAMA_Total'].dropna().values,
            info[4]['HAMA_Total'].dropna().values,
    ])
plt.title('HAMA')

plt.subplot(2,4,4)
vp().plot([
            info[0]['HAMD-17_Total'].dropna().values, 
            info[1]['HAMD-17_Total'].dropna().values,
            info[2]['HAMD-17_Total'].dropna().values,
            info[3]['HAMD-17_Total'].dropna().values,
            info[4]['HAMD-17_Total'].dropna().values,
    ])
plt.title('HAMD')

plt.subplot(2,4,5)
vp().plot([
            info[0]['BPRS_Total'].dropna().values, 
            info[1]['BPRS_Total'].dropna().values,
            info[2]['BPRS_Total'].dropna().values,
            info[3]['BPRS_Total'].dropna().values,
            info[4]['BPRS_Total'].dropna().values,
    ])
plt.title('BPRS')

plt.show()
