"""This script is designed to perform statistics of demographic information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr,spearmanr,kendalltau
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import os
from eslearn.utils.lc_read_write_mat import read_mat, write_mat

#%% ----------------------------------Our center 550----------------------------------
uid_path_550 = r'D:\WorkStation_2018\SZ_classification\Scale\selected_550.txt'
scale_path_550 = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
headmotion_file = r'D:\WorkStation_2018\SZ_classification\Scale\头动参数_1322.xlsx'

scale_data_550 = pd.read_excel(scale_path_550)
uid_550 = pd.read_csv(uid_path_550, header=None)

scale_selected_550 = pd.merge(uid_550, scale_data_550, left_on=0, right_on='folder', how='inner')
describe_bprs_550 = scale_selected_550.groupby('诊断')['BPRS_Total'].describe()
describe_age_550 = scale_selected_550.groupby('诊断')['年龄'].describe()
describe_duration_550 = scale_selected_550.groupby('诊断')['病程月'].describe()
describe_durgnaive_550 = scale_selected_550.groupby('诊断')['用药'].value_counts()
describe_sex_550 = scale_selected_550.groupby('诊断')['性别'].value_counts()

# Demographic
demographic_info_dataset1 = scale_selected_550[['folder', '诊断', '年龄', '性别', '病程月']]
headmotion = pd.read_excel(headmotion_file)
headmotion = headmotion[['Subject ID','mean FD_Power']]
demographic_info_dataset1 = pd.merge(demographic_info_dataset1, headmotion, left_on='folder', right_on='Subject ID', how='inner')
demographic_info_dataset1 = demographic_info_dataset1.drop(columns=['Subject ID'])

site_dataset1 = pd.DataFrame(np.zeros([len(demographic_info_dataset1),1]))
site_dataset1.columns = ['site']
demographic_dataset1_all = pd.concat([demographic_info_dataset1 , site_dataset1], axis=1)
demographic_dataset1_all.columns = ['ID','Diagnosis', 'Age', 'Sex', 'Duration', 'MeanFD', 'Site']
demographic_dataset1 = demographic_dataset1_all[['ID','Diagnosis', 'Age', 'Sex', 'MeanFD', 'Site']]
demographic_dataset1['Diagnosis'] = np.int32(demographic_dataset1['Diagnosis'] == 3)

# Duration and age
demographic_duration_dataset1 = demographic_dataset1_all[['Duration', 'Age']].dropna()
np.corrcoef(demographic_duration_dataset1['Duration'], demographic_duration_dataset1['Age'])
pearsonr(demographic_duration_dataset1['Duraton'], demographic_duration_dataset1['Age'])

#%% ----------------------------------BeiJing 206----------------------------------
uid_path_206 = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100.xlsx'
scale_path_206 = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100-WF.csv'
headmotion_file_206 = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\parameters\FD_power'
uid_to_remove = ['SZ010109','SZ010009']

scale_data_206 = pd.read_csv(scale_path_206)
scale_data_206 = scale_data_206.drop(np.array(scale_data_206.index)[scale_data_206['ID'].isin(uid_to_remove)])
scale_data_206['PANSStotal1'] = np.array([np.float64(duration) if duration.strip() !='' else 0 for duration in scale_data_206['PANSStotal1'].values])
Pscore = pd.DataFrame(scale_data_206[['P1', 'P2', 'P3', 'P4', 'P4', 'P5', 'P6', 'P7']].iloc[:106,:], dtype = np.float64)

Pscore = np.sum(Pscore, axis=1).describe()
Nscore = pd.DataFrame(scale_data_206[['N1', 'N2', 'N3', 'N4', 'N4', 'N5', 'N6', 'N7']].iloc[:106,:], dtype=np.float64)
Nscore = np.sum(Nscore, axis=1).describe()

Gscore = pd.DataFrame(scale_data_206[['G1', 'G2', 'G3', 'G4', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16']].iloc[:106,:])
Gscore = np.array(Gscore)
for i, itemi in enumerate(Gscore):
    for j, itemj in enumerate(itemi):
        print(itemj)
        if itemj.strip() != '':
            Gscore[i,j] = np.float64(itemj)
        else:
            Gscore[i, j] = np.nan
Gscore = pd.DataFrame(Gscore)      
Gscore = np.sum(Gscore, axis=1).describe()

describe_panasstotol_206 = scale_data_206.groupby('group')['PANSStotal1'].describe()
describe_age_206 = scale_data_206.groupby('group')['age'].describe()
scale_data_206['duration'] = np.array([np.float64(duration) if duration.strip() !='' else 0 for duration in scale_data_206['duration'].values])
describe_duration_206 = scale_data_206.groupby('group')['duration'].describe()
describe_sex_206 = scale_data_206.groupby('group')['sex'].value_counts()

# Demographic
uid = pd.DataFrame(scale_data_206['ID'])
uid['ID'] = uid['ID'].str.replace('NC','10');
uid['ID'] = uid['ID'].str.replace('SZ','20');
uid = pd.DataFrame(uid, dtype=np.int32)
demographic_info_dataset2 = scale_data_206[['group','age', 'sex']]
demographic_info_dataset2 = pd.concat([uid, demographic_info_dataset2], axis=1)

headmotion_name_dataset2 = os.listdir(headmotion_file_206)
headmotion_file_path_dataset2 = [os.path.join(headmotion_file_206, name) for name in headmotion_name_dataset2]

meanfd = []
for i, file in enumerate(headmotion_file_path_dataset2):
    fd = np.loadtxt(file)
    meanfd.append(np.mean(fd))  
meanfd_dataset2 = pd.DataFrame(meanfd)

headmotion_name_dataset2 = pd.Series(headmotion_name_dataset2)
headmotion_name_dataset2 = headmotion_name_dataset2.str.findall('(NC.*[0-9]\d*|SZ.*[0-9]\d*)')
headmotion_name_dataset2 = [str(id[0]) if id != [] else 0 for id in headmotion_name_dataset2]
headmotion_name_dataset2 = pd.DataFrame([''.join(id.split('_')) if id != 0 else '0' for id in headmotion_name_dataset2])
headmotion_name_dataset2[0] = headmotion_name_dataset2[0].str.replace('NC','10');
headmotion_name_dataset2[0] = headmotion_name_dataset2[0].str.replace('SZ','20');
headmotion_name_dataset2 = pd.DataFrame(headmotion_name_dataset2, dtype=np.int32)
headmotion_name_dataset2 = pd.concat([headmotion_name_dataset2, meanfd_dataset2], axis=1) 
headmotion_name_dataset2.columns = ['ID','meanFD']

demographic_dataset2 = pd.merge(demographic_info_dataset2, headmotion_name_dataset2, left_on='ID', right_on='ID', how='left')

site_dataset2 = pd.DataFrame(np.ones([len(demographic_dataset2),1]))
site_dataset2.columns = ['site']

demographic_dataset2 = pd.concat([demographic_dataset2, site_dataset2], axis=1)
demographic_dataset2.columns = ['ID', 'Diagnosis', 'Age', 'Sex', 'MeanFD', 'Site']
demographic_dataset2['Diagnosis'] = np.int32(demographic_dataset2['Diagnosis'] == 1)

duration_dataset2 = pd.concat([uid, scale_data_206['duration']], axis=1)
demographic_duration_dataset2 = pd.merge(duration_dataset2, demographic_dataset2, left_on='ID', right_on='ID')
demographic_duration_dataset2 = demographic_duration_dataset2.iloc[:106,:]
pearsonr(demographic_duration_dataset2['duration'], demographic_duration_dataset2['Age'])

#%% -------------------------COBRE----------------------------------
# Inputs
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC_COBRE'  # all mat files directory
scale = r'H:\Data\精神分裂症\COBRE\COBRE_phenotypic_data.csv'  # whole scale path
headmotion_file_COBRE = r'D:\WorkStation_2018\SZ_classification\Data\headmotion\cobre\HeadMotion.tsv'
duration_COBRE = r'D:\WorkStation_2018\SZ_classification\Scale\COBRE_duration.xlsx'

# Transform the .mat files to one .npy file
allmatname = os.listdir(matroot)

# Give labels to each subject, concatenate at the first column
allmatname = pd.DataFrame(allmatname)
allsubjname = allmatname.iloc[:,0].str.findall(r'[1-9]\d*')
allsubjname = pd.DataFrame([name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep=',',dtype='str')
print(scale_data)
diagnosis = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')[['ID','Subject Type']]
scale_data = pd.merge(allsubjname,scale_data,left_on=0,right_on='ID')

diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Control'] = 0
diagnosis['Subject Type'][diagnosis['Subject Type'] == 'Patient'] = 1
include_loc = diagnosis['Subject Type'] != 'Disenrolled'
diagnosis = diagnosis[include_loc.values]
allsubjname = allsubjname[include_loc.values]
scale_data_COBRE = pd.merge(allsubjname, scale_data, left_on=0, right_on=0, how='inner').iloc[:,[0,1,2,3,5]]
scale_data_COBRE['Gender'] = scale_data_COBRE['Gender'].str.replace('Female', '0')
scale_data_COBRE['Gender'] = scale_data_COBRE['Gender'].str.replace('Male', '1')
scale_data_COBRE['Subject Type'] = scale_data_COBRE['Subject Type'].str.replace('Patient', '1')
scale_data_COBRE['Subject Type'] = scale_data_COBRE['Subject Type'].str.replace('Control', '0')
scale_data_COBRE = pd.DataFrame(scale_data_COBRE, dtype=np.float64)

describe_age_COBRE = scale_data_COBRE.groupby('Subject Type')['Current Age'].describe()
describe_sex_COBRE = scale_data_COBRE.groupby('Subject Type')['Gender'].value_counts()

headmotion_COBRE = pd.read_csv(headmotion_file_COBRE,sep='\t', index_col=False)
headmotion_COBRE = headmotion_COBRE[['Subject ID', 'mean FD_Power']]

scale_data['ID'] = pd.DataFrame(scale_data['ID'], dtype=np.int32)
demographic_COBRE = pd.merge(scale_data, headmotion_COBRE, left_on='ID', right_on='Subject ID', how='inner')
demographic_COBRE = demographic_COBRE[['ID', 'Subject Type', 'Current Age', 'Gender', 'mean FD_Power']]

site_COBRE = pd.DataFrame(np.ones([len(demographic_COBRE),1]) + 1)
site_COBRE.columns = ['site']

demographic_COBRE = pd.concat([demographic_COBRE, site_COBRE], axis=1).drop([70,82])
demographic_COBRE['Gender'] = demographic_COBRE['Gender']  == 'Male'
demographic_COBRE[['Current Age', 'Gender']]  = np.int32(demographic_COBRE[['Current Age', 'Gender']] )
demographic_COBRE.columns = ['ID', 'Diagnosis', 'Age', 'Sex', 'MeanFD', 'Site']
demographic_COBRE['Diagnosis'] = np.int32(demographic_COBRE['Diagnosis'] == 'Patient')

duration_COBRE = pd.read_excel(duration_COBRE)
duration_COBRE = duration_COBRE.iloc[:,[0,1,2]]
duration_COBRE = duration_COBRE.dropna()
duration_COBRE = pd.DataFrame(duration_COBRE, dtype=np.int32)
duration_COBRE[duration_COBRE == 9999] = None
duration_COBRE = duration_COBRE.dropna()
duration_COBRE['duration'] = duration_COBRE.iloc[:,1] - duration_COBRE.iloc[:,2]
duration_COBRE['duration']  =duration_COBRE['duration'] * 12
duration_COBRE.columns = ['ID', 'Age', 'Onset_age', 'Duration']
demographic_druation_COBRE = pd.merge(demographic_COBRE, duration_COBRE, left_on='ID', right_on='ID', how='inner')

# Correattion of duration and age
pearsonr(demographic_druation_COBRE['Duration'], demographic_druation_COBRE ['Age_x'])

#%% -------------------------UCLA----------------------------------
matroot = r'D:\WorkStation_2018\SZ_classification\Data\SelectedFC_UCLA'
scale = r'H:\Data\精神分裂症\ds000030\schizophrenia_UCLA_restfmri\participants.tsv'
headmotion_UCAL = r'D:\WorkStation_2018\SZ_classification\Data\headmotion\ucal\HeadMotion.tsv'
headmotion_UCAL_rest = r'D:\WorkStation_2018\SZ_classification\Data\headmotion\ucal\HeadMotion_rest.tsv'

allmatname = os.listdir(matroot)
allmatname = pd.DataFrame(allmatname)
allsubjname = allmatname.iloc[:,0].str.findall(r'[1-9]\d*')
allsubjname = pd.DataFrame(['sub-' + name[0] for name in allsubjname])
scale_data = pd.read_csv(scale,sep='\t')
scale_data_UCAL = pd.merge(allsubjname,scale_data,left_on=0,right_on='participant_id')
scale_data_UCAL['diagnosis'][scale_data_UCAL['diagnosis'] == 'CONTROL']=0
scale_data_UCAL['diagnosis'][scale_data_UCAL['diagnosis'] == 'SCHZ']=1
scale_data_UCAL['participant_id'] = scale_data_UCAL['participant_id'].str.replace('sub-', '')
scale_data_UCAL = pd.merge(allsubjname,scale_data_UCAL, left_on=0, right_on=0, how='inner')
scale_data_UCAL = scale_data_UCAL.iloc[:,[1,2,3,4]]
scale_data_UCAL['gender'] = scale_data_UCAL['gender'].str.replace('M', '1')
scale_data_UCAL['gender'] = scale_data_UCAL['gender'].str.replace('F', '0')
scale_data_UCAL = pd.DataFrame(scale_data_UCAL, dtype=np.float64)
describe_age_UCAL = scale_data_UCAL.groupby('diagnosis')['age'].describe()
describe_sex_UCAL = scale_data_UCAL.groupby('diagnosis')['gender'].value_counts()

headmotion1_UCAL = pd.read_csv(headmotion_UCAL, sep='\t', index_col=False)[['Subject ID','mean FD_Power']]
headmotion1_UCAL['Subject ID'] = headmotion1_UCAL['Subject ID'].str.findall(r'[1-9]\d*')
headmotion1_UCAL['Subject ID'] = [np.int32(idx[0]) for idx in headmotion1_UCAL['Subject ID']]
headmotion2_UCAL = pd.read_csv(headmotion_UCAL_rest, sep='\t', index_col=False)[['Subject ID', 'mean FD_Power']]
headmotion_UCAL = pd.concat([headmotion1_UCAL, headmotion2_UCAL])

demographic_UCAL = pd.merge(scale_data_UCAL, headmotion_UCAL, left_on='participant_id', right_on='Subject ID')
demographic_UCAL = demographic_UCAL.drop(columns=['Subject ID'])

site_UCAL = pd.DataFrame(np.ones([len(demographic_UCAL), 1])+2)

demographic_UCAL = pd.concat([demographic_UCAL, site_UCAL], axis=1)
demographic_UCAL.columns = ['ID', 'Diagnosis', 'Age', 'Sex', 'MeanFD', 'Site']


#%% Load all npy
dataset1_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy'
dataset2_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy'
dataset_COBRE_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_COBRE.npy'
dataset_UCAL_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_UCLA.npy'

dataset1 = np.load(dataset1_file)
dataset2 = np.load(dataset2_file)
dataset_COBRE = np.load(dataset_COBRE_file)
dataset_UCAL = np.load(dataset_UCAL_file)

#%% Sort demogrphic data with npy
demographic_all = pd.concat([demographic_dataset1, demographic_dataset2, demographic_COBRE, demographic_UCAL], axis=0)
dataset_all = np.concatenate([dataset1, dataset2, dataset_COBRE, dataset_UCAL], axis=0)
write_mat(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat',dataset_name='dataset_all', dataset=dataset_all)
dataset_all = pd.DataFrame(dataset_all)

demographic_all = pd.merge(dataset_all, demographic_all, left_on=0, right_on='ID', how='inner')[['ID', 'Diagnosis', 'Age', 'Sex', 'MeanFD', 'Site']]

demographic_all.to_excel(r'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx', index=None)


#%% Correlation of duration and age
duration_age_dataset1 = demographic_duration_dataset1[['病程月','Age']].values
duration_age_dataset2 = demographic_duration_dataset2[['duration','Age']].values
duration_age_dataset_COBRE = demographic_druation_COBRE[['Duration','Age_x']].values
duration_age_all = np.concatenate([duration_age_dataset1, duration_age_dataset2, duration_age_dataset_COBRE])
group = np.concatenate([np.ones(len(duration_age_dataset1,))+100, np.ones(len(duration_age_dataset2,))+1000, np.ones(len(duration_age_dataset_COBRE,))+10000])
duration_age_all = pd.concat([pd.DataFrame(duration_age_all), pd.DataFrame(group)], axis=1)
duration_age_all.columns = ['Duration', 'Age', 'Dataset']
duration_age_all['Dataset'] = [str(int(g)) for g in duration_age_all['Dataset']]
duration_age_all['Dataset'] = duration_age_all['Dataset'].str.replace('101','Dataset 1')
duration_age_all['Dataset'] = duration_age_all['Dataset'].str.replace('1001','Dataset 2')
duration_age_all['Dataset'] = duration_age_all['Dataset'].str.replace('10001','Dataset 3')

# plot scatter of duration and age
pearsonr(duration_age_all['Duration'], duration_age_all['Age'])
g = sns.jointplot('Duration', 'Age', data=duration_age_all,col='Dataset',
                  kind="reg", truncate=False, height=7)

ax = sn.lmplot('Duration', 'Age', data=duration_age_all, col='Dataset')
ax = sn.lmplot('Duration', 'Age', data=duration_age_all)
ax.set_titles(fontsize=20, fontweight='bold')

plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\correlation_duration_age_all.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)


# sns.jointplot(duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 1'], duration_age_all['Age'][duration_age_all['Dataset']=='Dataset 1'],
#               kind="reg", truncate=False, height=7, xlim=(0, duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 1'].max()))

# sns.jointplot(duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 2'], duration_age_all['Age'][duration_age_all['Dataset']=='Dataset 2'],
#               kind="reg", truncate=False, height=7, xlim=(0, duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 2'].max()))

# sns.jointplot(duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 3'], duration_age_all['Age'][duration_age_all['Dataset']=='Dataset 3'],
#               kind="reg", truncate=False, height=7, xlim=(0, duration_age_all['Duration'][duration_age_all['Dataset']=='Dataset 3'].max()))