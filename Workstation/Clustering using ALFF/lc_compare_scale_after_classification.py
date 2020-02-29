# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:22:19 2018
应用聚类模型，对测试集分类后，比较各个类的量表差异
@author: lenovo
"""
#
import sys
import pandas as pd
import numpy as np
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\Statistic')
from lc_chi2 import lc_chi2
from lc_ttest2 import ttest2
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

# input
scale_file=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\REST-meta-MDD-PhenotypicData_WithHAMDSubItem_S20.xlsx'

pred_label_file=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\predict_label_multilabel.xlsx'

# load file
scale=pd.read_excel(scale_file)

pred_label=pd.read_excel(pred_label_file,header=None)
pred_label.columns=['predict_label']

# concat
scale=pd.concat([scale,pred_label],axis=1)

# dropna and others
scale=scale.mask(scale==-9999,None)
scale=scale.mask(scale=='[]',None)
scale=scale.dropna()

# save
#scale.to_excel(r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\multilabel\scale_predct_label.xlsx',index=False)

#==============================================================================
# statistic

# sex
data_sex=[list(scale[scale['predict_label']==1]['性别'].value_counts()),
          list(scale[scale['predict_label']==2]['性别'].value_counts()),
          list(scale[scale['predict_label']==3]['性别'].value_counts()),
          list(scale[scale['predict_label']==4]['性别'].value_counts()),
          list(scale[scale['predict_label']==5]['性别'].value_counts())]

chi_sex,p_sex=lc_chi2(data_sex)

# age
model_age = ols('年龄~predict_label',scale).fit()
anova_age = sm.stats.anova_lm(model_age)
print(anova_age)

p_age=anova_age.iloc[0,-1]
F_age=anova_age.iloc[0,-2]

# post_hoc
p=np.ones([5,5])
for i in np.arange(1,6,1):
    for j in np.arange(1,6,1):
        t,p[i-1,j-1]=ttest2(scale[scale['predict_label']==i]['年龄'].values,
                   scale[scale['predict_label']==j]['年龄'].values,
                   method='independent')
        
        
# hot map
f, (ax1) = plt.subplots(figsize=(6,6),nrows=1)

#sns.heatmap(x, annot=True, ax=ax1,cmap='rainbow',center=0)#cmap='rainbow'
sns.heatmap(p,ax=ax1,
            annot=True,annot_kws={'size':9,'weight':'normal', 'color':'w'},fmt='.2f',
            cmap='RdBu',
            linewidths = 0.05, linecolor= 'w',
            mask=p>0.05)

ax1.set_title('P values of age')
ax1.set_xticklabels(labels=[1,2,3,4,5],fontsize=15)
ax1.set_yticklabels(labels=[1,2,3,4,5],fontsize=15)

# education
model_edu = ols('教育年限~C(predict_label)',scale).fit()
anova_edu = anova_lm(model_edu)
print(anova_edu)

p_edu=anova_edu.iloc[0,-1]
F_edu=anova_edu.iloc[0,-2]

# medication
data_med=[list(scale[scale['predict_label']==1]['是否正在用药'].value_counts()),
          list(scale[scale['predict_label']==2]['是否正在用药'].value_counts()),
          list(scale[scale['predict_label']==3]['是否正在用药'].value_counts()),
          list(scale[scale['predict_label']==4]['是否正在用药'].value_counts()),
          list(scale[scale['predict_label']==5]['是否正在用药'].value_counts())]

chi_med,p_med=lc_chi2(data_med)

# HAMD
# 先把HAMD变为数据类型
scale['HAMD']=scale['HAMD'].astype('float32')

model_HAMD = ols("HAMD~C(predict_label)",scale).fit()
anova_HAMD = anova_lm(model_HAMD)

p_HAMD=anova_HAMD.iloc[0,-1]
f_HAMD=anova_HAMD.iloc[0,-2]

#==============================================================================
# HAMA
# 先把HAMD变为数据类型
scale['HAMA']=scale['HAMA'].astype('float32')

model_HAMA = ols("HAMA~C(predict_label)",scale).fit()
anova_HAMA = anova_lm(model_HAMA)

p_HAMA=anova_HAMA.iloc[0,-1]
f_HAMA=anova_HAMA.iloc[0,-2]



# post_hoc
p=np.ones([5,5])
for i in np.arange(1,6,1):
    for j in np.arange(1,6,1):
        t,p[i-1,j-1]=ttest2(scale[scale['predict_label']==i]['HAMA'].values,
                   scale[scale['predict_label']==j]['HAMA'].values,
                   method='independent')
        
        
# hot map
f, (ax1) = plt.subplots(figsize=(6,6),nrows=1)

#sns.heatmap(x, annot=True, ax=ax1,cmap='rainbow',center=0)#cmap='rainbow'
sns.heatmap(p,ax=ax1,
            annot=True,annot_kws={'size':9,'weight':'normal', 'color':'w'},fmt='.2f',
            cmap='RdBu',
            linewidths = 0.05, linecolor= 'w',
            mask=p>0.05)

ax1.set_title('P values of HAMA')
ax1.set_xticklabels(labels=[1,2,3,4,5],fontsize=15)
ax1.set_yticklabels(labels=[1,2,3,4,5],fontsize=15)