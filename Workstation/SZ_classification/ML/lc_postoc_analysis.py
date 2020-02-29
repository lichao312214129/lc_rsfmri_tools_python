# -*- coding: utf-8 -*-
"""
This script is used to perform post-hoc analysis and visualization: 
the classification performance of subsets (only for SZ).
"""

#%%
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import pickle

plt.rcParams['savefig.dpi'] = 1200

#%% Inputs
scale_550_file = r'D:\WorkStation_2018\SZ_classification\Scale\10-24大表.xlsx'
scale_206_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_NC_108_100-WF.csv'
scale_206_drug_file = r'D:\WorkStation_2018\SZ_classification\Scale\北大精分人口学及其它资料\SZ_109_drug.xlsx'
classification_results_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results.npy'

# Load scales and results
scale_550 = pd.read_excel(scale_550_file)
scale_206 = pd.read_csv(scale_206_file)
scale_206_drug = pd.read_excel(scale_206_drug_file)
results = pd.DataFrame(np.load(classification_results_file))
print(f'results = {results}')

# Proprocess
scale_206['ID'] = scale_206['ID'].str.replace('NC','1')
scale_206['ID'] = scale_206['ID'].str.replace('SZ','2')
scale_206['ID'] = np.int32(scale_206['ID'])
scale_550['folder'] = np.int32(scale_550['folder'])

scale_206_drug['P0001'] = scale_206_drug['P0001'].str.replace('NC','1')
scale_206_drug['P0001'] = scale_206_drug['P0001'].str.replace('SZ','2')
scale_206_drug['P0001'] = np.int32(scale_206_drug['P0001'])

# Filter subjects that have .mat files
scale_550_selected = pd.merge(results, scale_550, left_on=0, right_on='folder', how='inner')
scale_206_selected = pd.merge(results, scale_206, left_on=0, right_on='ID', how='inner')
scale_206_selected = pd.merge(scale_206_selected, scale_206_drug, left_on=0, right_on='P0001', how='inner')

#%% ---------------------------------Calculate performance for SZ sub-groups-------------------------------
## Step 1: Dataset1
duration = 18
data_firstepisode_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['首发'] == 1) & (scale_550_selected['病程月'] <= duration)]
data_notfirstepisode_550 = scale_550_selected[(scale_550_selected['诊断']==3) & ((scale_550_selected['首发'] == 0) | (scale_550_selected['病程月'] > duration))]  # Including the persistent patients

data_shortduration_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['病程月'] <= duration)]
data_longduration_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['病程月'] > duration)]

age_sz = scale_550_selected['年龄'][scale_550_selected['诊断']==3]
data_young_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (age_sz <= np.median(age_sz))]
data_old_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (age_sz > np.median(age_sz))]

onsetage_sz = scale_550_selected['Age_of_first_episode'][scale_550_selected['诊断']==3].dropna()
ind_young_onsetage_550 = onsetage_sz.index[onsetage_sz.values <= np.median(onsetage_sz)]
ind_old_onsetage_550 = onsetage_sz.index[onsetage_sz.values > np.median(onsetage_sz)]
data_young_ageoffirst_episode_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_young_onsetage_550]
data_old_onsetage_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_old_onsetage_550]

data_male_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['性别'] == 1)]
data_female_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['性别'] == 2)]

data_unmedicated_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] == 0)]
data_medicated_550 = scale_550_selected[(scale_550_selected['诊断']==3) & (scale_550_selected['用药'] == 1)]

# Frist episode and nerver medicated
data_unmedicated_and_firstepisode_550 = scale_550_selected[(scale_550_selected['诊断']==3) & 
                                            (scale_550_selected['首发'] == 1) &
                                            (scale_550_selected['病程月'] <= duration) &
                                            (scale_550_selected['用药'] == 0)]

data_unmedicated_and_chronic_550 = scale_550_selected[(scale_550_selected['诊断']==3) & 
                                            (scale_550_selected['病程月'] > duration) &
                                            (scale_550_selected['用药'] == 0)]
# data_unmedicated_550['folder'].to_csv(r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Scale\unmedicated_63.txt', index=False)

headmotion_sz = scale_550_selected['头动（7000）'][scale_550_selected['诊断']==3].dropna()
ind_less = list(headmotion_sz.index[headmotion_sz.values <= np.median(headmotion_sz.values)])
ind_greater = list(headmotion_sz.index[headmotion_sz.values > np.median(headmotion_sz.values)])
data_lessheadmotion_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_less]
data_greaterheadmotion_sz_550 = scale_550_selected[scale_550_selected['诊断']==3].loc[ind_greater]

## Calculating Accuracy
acc_firstepisode_550 = np.sum(data_firstepisode_550[1]-data_firstepisode_550[3]==0)/len(data_firstepisode_550)
acc_notfirstepisode_550 = np.sum(data_notfirstepisode_550[1]-data_notfirstepisode_550[3]==0)/len(data_notfirstepisode_550)

acc_shortduration_550 = np.sum(data_shortduration_550[1]-data_shortduration_550[3]==0)/len(data_shortduration_550)
acc_longduration_550 = np.sum(data_longduration_550[1]-data_longduration_550[3]==0)/len(data_longduration_550)

acc_young_550 = np.sum(data_young_550[1]-data_young_550[3]==0)/len(data_young_550)
acc_old_550 = np.sum(data_old_550[1]-data_old_550[3]==0)/len(data_old_550)

acc_young_onsetage_550 = np.sum(data_young_ageoffirst_episode_550[1]-data_young_ageoffirst_episode_550[3]==0)/len(data_young_ageoffirst_episode_550)
acc_old_onsetage_550 = np.sum(data_old_onsetage_550[1]-data_old_onsetage_550[3]==0)/len(data_old_onsetage_550)

acc_male_550 = np.sum(data_male_550[1]-data_male_550[3]==0)/len(data_male_550)
acc_female_550 = np.sum(data_female_550[1]-data_female_550[3]==0)/len(data_female_550)

acc_unmedicated_550 = np.sum(data_unmedicated_550[1]-data_unmedicated_550[3]==0)/len(data_unmedicated_550)
acc_medicated_550 = np.sum(data_medicated_550[1]-data_medicated_550[3]==0)/len(data_medicated_550)

acc_unmedicated_and_firstepisode_550 = np.sum(data_unmedicated_and_firstepisode_550[1]-data_unmedicated_and_firstepisode_550[3]==0) / len(data_unmedicated_and_firstepisode_550)
acc_unmedicated_and_chronic_550 = np.sum(data_unmedicated_and_chronic_550[1]-data_unmedicated_and_chronic_550[3]==0) / len(data_unmedicated_and_chronic_550)

acc_lessheadmotion_550 = np.sum(data_lessheadmotion_550[1]-data_lessheadmotion_550[3]==0)/len(data_lessheadmotion_550)
acc_greaterheadmotion_550 = np.sum(data_greaterheadmotion_sz_550[1]-data_greaterheadmotion_sz_550[3]==0)/len(data_greaterheadmotion_sz_550)

print(f'Accuracy of firstepisode in dataset550 = {acc_firstepisode_550}')
print(f'Accuracy of none-firstepisode in dataset550 = {acc_notfirstepisode_550}')

print(f'Accuracy of shortduration in dataset550 = {acc_shortduration_550}')
print(f'Accuracy of longduration in dataset550 = {acc_longduration_550}')

print(f'Accuracy of young in dataset550 = {acc_young_550}')
print(f'Accuracy of old in dataset550 = {acc_old_550}')

print(f'Accuracy of young onsetage of 550 = {acc_young_onsetage_550}')
print(f'Accuracy of old onsetage of 550 = {acc_old_onsetage_550}')

print(f'Accuracy of male of 550 = {acc_male_550}')
print(f'Accuracy of female of 550 = {acc_female_550}')

print(f'Accuracy of lessheadmotion in dataset550 = {acc_lessheadmotion_550}')
print(f'Accuracy of greaterheadmotion in dataset550 = {acc_greaterheadmotion_550}')

print(f'Accuracy of unmedicated in dataset550 = {acc_unmedicated_550}')
print(f'Accuracy of medicated in dataset550 = {acc_medicated_550}')

print(f'Accuracy of unmedicated and firstepisode in dataset550 = {acc_unmedicated_and_firstepisode_550}')
print(f'Accuracy of medicated and chronic in dataset550 = {acc_unmedicated_and_chronic_550}')
print('-'*50)

# Step 2: Dataset 2
## Preprocessing
scale_206_selected['duration'] = [np.int32(duration) if duration != ' ' else 10000 for duration in scale_206_selected['duration']]
scale_206_selected['firstepisode'] = [np.int32(firstepisode) if firstepisode != ' ' else 10000 for firstepisode in scale_206_selected['firstepisode']]
scale_206_selected['CPZ_eq'] = [np.int32(duration) if duration != ' ' else 0 for duration in scale_206_selected['CPZ_eq']]
scale_206_selected['onsetage'] = [np.int32(duration) if duration != ' ' else 0 for duration in scale_206_selected['onsetage']]

## Filter subgroups
data_firstepisode_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['firstepisode'] == 1) & (scale_206_selected['duration'] <= duration)]
data_notfirstepisode_206 = scale_206_selected[(scale_206_selected['group']==1) & ((scale_206_selected['firstepisode'] == 0) | (scale_206_selected['duration'] > duration))]

data_shortduration_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['duration'] <= duration)]
data_longduration_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['duration'] > duration)]

age_sz_206 = scale_206_selected['age'][scale_206_selected['group']==1]
data_young_206 = scale_206_selected[(scale_206_selected['group']==1) & (age_sz_206 <= np.median(age_sz_206))]
data_old_206 = scale_206_selected[(scale_206_selected['group']==1) & (age_sz_206 > np.median(age_sz_206))]

onsetage = scale_206_selected['onsetage'][scale_206_selected['group']==1]
data_young_onsetage_206 = scale_206_selected[(scale_206_selected['group']==1) & (onsetage <= np.median(onsetage))]
data_old_onsetage_206 = scale_206_selected[(scale_206_selected['group']==1) & (onsetage > np.median(onsetage))]

data_male_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['sex'] == 1)]
data_female_206 = scale_206_selected[(scale_206_selected['group']==1) & (scale_206_selected['sex'] == 0)]

CPZ_eq = scale_206_selected['CPZ_eq'][scale_206_selected['group']==1]
data_drugless_206 = scale_206_selected[(scale_206_selected['group']==1) & (CPZ_eq <= np.median(CPZ_eq))]
data_drugmore_206 = scale_206_selected[(scale_206_selected['group']==1) & (CPZ_eq > np.median(CPZ_eq))]

## Calculating acc
acc_firstepisode_206 = np.sum(data_firstepisode_206[1]-data_firstepisode_206[3]==0)/len(data_firstepisode_206)
acc_notfirstepisode_206 = np.sum(data_notfirstepisode_206[1]-data_notfirstepisode_206[3]==0)/len(data_notfirstepisode_206)

acc_shortduration_206 = np.sum(data_shortduration_206[1]-data_shortduration_206[3]==0)/len(data_shortduration_206)
acc_longduration_206 = np.sum(data_longduration_206[1]-data_longduration_206[3]==0)/len(data_longduration_206)

acc_young_206 = np.sum(data_young_206[1]-data_young_206[3]==0)/len(data_young_206)
acc_old_206 = np.sum(data_old_206[1]-data_old_206[3]==0)/len(data_old_206)

acc_young_onsetage_206 = np.sum(data_young_onsetage_206[1]-data_young_onsetage_206[3]==0)/len(data_young_onsetage_206)
acc_old_onsetage_206 = np.sum(data_old_onsetage_206[1]-data_old_onsetage_206[3]==0)/len(data_old_onsetage_206)

acc_male_206 = np.sum(data_male_206[1]-data_male_206[3]==0)/len(data_male_206)
acc_female_206 = np.sum(data_female_206[1]-data_female_206[3]==0)/len(data_female_206)

acc_drugless_206 = np.sum(data_drugless_206[1]-data_drugless_206[3]==0)/len(data_drugless_206)
acc_drugmore_206 = np.sum(data_drugmore_206[1]-data_drugmore_206[3]==0)/len(data_drugmore_206)

##
print(f'Accuracy of first episode of 206 = {acc_firstepisode_206}')
print(f'Accuracy of not first episode of 206 = {acc_notfirstepisode_206}')

print(f'Accuracy of shortduration of 206 = {acc_shortduration_206}')
print(f'Accuracy of longduration of 206 = {acc_longduration_206}')

print(f'Accuracy of young of 206 = {acc_young_206}')
print(f'Accuracy of old of 206 = {acc_old_206}')

print(f'Accuracy of young onsetage of 206 = {acc_young_onsetage_206}')
print(f'Accuracy of old onsetage of 206 = {acc_old_onsetage_206}')

print(f'Accuracy of male of 206 = {acc_male_206}')
print(f'Accuracy of female of 206 = {acc_female_206}')

print(f'Accuracy of drugless of 206 = {acc_drugless_206}')
print(f'Accuracy of drugmore of 206 = {acc_drugmore_206}')

#%% -------------------------------------Visualization-----------------------------------------------
# Load all resluts
with open(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_pooling', 'rb') as f1:
    results_pooling = pickle.load(f1)
accuracy_pooling = results_pooling['accuracy']
sensitivity_pooling = results_pooling['sensitivity']
specificity_pooling = results_pooling['specificity']
AUC_pooling = results_pooling['AUC']
performances_pooling = [accuracy_pooling, sensitivity_pooling, specificity_pooling, AUC_pooling]
performances_pooling = pd.DataFrame(performances_pooling)
performances_pooling.to_csv(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_pooling.txt', index=False, header=False)

with open(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_leave_one_site_cv', 'rb') as f2:
    results_leave_one_site_cv = pickle.load(f2)
accuracy_leave_one_site_cv = results_leave_one_site_cv['accuracy']
sensitivity_leave_one_site_cv = results_leave_one_site_cv['sensitivity']
specificity_leave_one_site_cv = results_leave_one_site_cv['specificity']
AUC_leave_one_site_cv = results_leave_one_site_cv['AUC']
performances_leave_one_site_cv = [accuracy_leave_one_site_cv, sensitivity_leave_one_site_cv, specificity_leave_one_site_cv, AUC_leave_one_site_cv]
performances_leave_one_site_cv = pd.DataFrame(performances_leave_one_site_cv)
performances_leave_one_site_cv.to_csv(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_leave_one_site_cv.txt', index=False, header=False)

with open(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_unmedicated_and_firstepisode_550.npy', 'rb') as f3:
    results_unmedicated = pickle.load(f3)
accuracy_unmedicated = results_unmedicated['accuracy']
sensitivity_unmedicated = results_unmedicated['sensitivity']
specificity_unmedicated = results_unmedicated['specificity']
AUC_unmedicated = results_unmedicated['AUC']
performances_unmedicated = [accuracy_unmedicated, sensitivity_unmedicated, specificity_unmedicated, AUC_unmedicated]
performances_unmedicated = pd.DataFrame(performances_unmedicated)
performances_unmedicated.to_csv(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_unmedicated.txt', index=False, header=False)

# performances_pooling_file = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\performances_pooling.txt';
# performances_leave_one_site_cv_file = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\performances_leave_one_site_cv.txt';
# performances_unmedicated_file = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\performances_unmedicated.txt';

# Save weights to .mat file for visulazation using MATLAB.
import scipy.io as io

weight_pooling_1d = np.mean(np.squeeze(np.array(results_pooling['coef'])), axis=0)
weight_leave_one_out_cv_1d = np.mean(np.squeeze(np.array(results_leave_one_site_cv['coef'])), axis=0)
weight_unmedicated_1d = np.mean(np.squeeze(np.array(results_unmedicated['coef'])), axis=0)

mask = np.triu(np.ones([246, 246]), 1) == 1

weight_pooling = np.zeros([246, 246])
weight_leave_one_out_cv = np.zeros([246, 246])
weight_unmedicated = np.zeros([246, 246])

weight_pooling[mask] = weight_pooling_1d
weight_leave_one_out_cv[mask] = weight_leave_one_out_cv_1d
weight_unmedicated[mask] = weight_unmedicated_1d

weight_pooling = weight_pooling + weight_pooling.T;
weight_leave_one_out_cv = weight_leave_one_out_cv + weight_leave_one_out_cv.T
weight_unmedicated = weight_unmedicated + weight_unmedicated.T

io.savemat(r'D:\WorkStation_2018\SZ_classification\Figure\weights.mat',
           {'weight_pooling': weight_pooling,
            'weight_leave_one_out_cv': weight_leave_one_out_cv,
            'weight_unmedicated': weight_unmedicated})



# Bar: performances in the whole Dataset.
import seaborn as sns
plt.figure(figsize=(20,30))
all_mean = np.concatenate([np.mean(performances_pooling.values,1), np.mean(performances_leave_one_site_cv.values,1), np.mean(performances_unmedicated.values,1)])
error = np.concatenate([np.std(performances_pooling.values, 1), np.std(performances_leave_one_site_cv.values, 1), np.std(performances_unmedicated.values, 1)])

plt.subplot(3, 1, 1)
color = ['darkturquoise'] * 4 +  ['paleturquoise'] * 4 + ['lightblue'] * 4
plt.bar(np.arange(0,len(all_mean)), all_mean, yerr = error, 
        capsize=5, linewidth=2, color=color)
plt.tick_params(labelsize=20)
plt.xticks(np.arange(0,len(all_mean)), ['Accuracy', 'Sensitivity', 'Sensitivity', 'AUC'] * 3, fontsize=20, rotation=45)
plt.title('Classification performances', fontsize=25, fontweight='bold')

y_major_locator=MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='y')
plt.fill_between(np.linspace(-0.4,3.4), 1.01, 1.05, color='darkturquoise')
plt.fill_between(np.linspace(3.6, 7.4), 1.01, 1.05, color='paleturquoise')
plt.fill_between(np.linspace(7.6, 11.4), 1.01, 1.05, color='lightblue')
               
# Bar: Dataset 1
plt.subplot(3,1,2)
barcont_550 = [acc_firstepisode_550, acc_notfirstepisode_550,
        acc_shortduration_550, acc_longduration_550, 
        acc_young_550, acc_old_550,
        acc_young_onsetage_550, acc_old_onsetage_550, 
        acc_male_550, acc_female_550, 
        acc_medicated_550, acc_unmedicated_550, 
        acc_unmedicated_and_firstepisode_550, acc_unmedicated_and_chronic_550]
label_550 = ['First episode', 'Not first episode', 'Short duration', 'Long duration',
         'Young', 'Elder', 'Young onset age','Elder onset age', 
        'Male','Female', 'Medicated', 'Unmedicated', 'First episode unmedicated', 'Chronic unmedicated']
    
# Bar: Dataset 2
barcont_206 = [acc_firstepisode_206, acc_notfirstepisode_206,
        acc_shortduration_206, acc_longduration_206, 
        acc_young_206, acc_old_206,
        acc_young_onsetage_206, acc_old_onsetage_206,
        acc_male_206, acc_female_206, acc_drugless_206, acc_drugmore_206]
label_206 = ['First episode', 'Not first episode', 'Short duration', 'Long duration',
         'Young', 'Elder', 'Young onset age','Elder onset age', 
        'Male','Female', 'Larger dosage (CPZ equivalent)', 'Small dosage (CPZ equivalent)']

## Plot
barcont_all = barcont_206 + barcont_550
label_all = label_206 + label_550 
color = ['lightblue' for i in range(12)] + ['darkturquoise' for i in range(14)]
h = plt.barh(np.arange(0,len(barcont_all)), barcont_all,color=color)
plt.legend(h, ['Dataset 1','Dataset 2'], fontsize=20)  # TODO
plt.tick_params(labelsize=20)
plt.yticks(np.arange(0,len(barcont_all)), label_all, fontsize=20, rotation=0)
plt.title('Accuracy of each subgroup of SZ \nin Dataset 1 and Dateset 2', fontsize=25, fontweight='bold')

x_major_locator=MultipleLocator(0.1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Accuracy', fontsize=25)
plt.grid(axis='x')
xticks = ax.get_xticks()
yticks = ax.get_yticks()
for i, (y,x) in enumerate(zip(yticks, barcont_all)):
    if np.isin(i, (24)):
        plt.text(x+0.01, y-0.3, '%.2f' % x,fontsize=20, color='r',fontweight='bold')
    else:
        plt.text(x+0.01, y-0.3, '%.2f' % x,fontsize=15)

## Bootstrap analysis of unmedicated and first episode: Dataset 1: Randomly selectting the subgroup of SZ: Bootstrap
plt.subplot(3,1,3)
data_all_550 = scale_550_selected[scale_550_selected['诊断']==3]
print(f'Shape of unmedicated and first episode = {data_unmedicated_and_firstepisode_550.shape}')
acc_true = np.sum(data_unmedicated_and_firstepisode_550[1]-data_unmedicated_and_firstepisode_550[3]==0)/len(data_unmedicated_and_firstepisode_550)
acc_random = []
for i in range(1000):
    np.random.seed(i)
    randomloc = np.random.permutation(data_all_550.shape[0])
    randomloc = randomloc[:data_unmedicated_550.shape[0]]
    data_all_550_permutated = data_all_550.iloc[randomloc,:]
    acc_random.append(np.sum(data_all_550_permutated[1]-data_all_550_permutated[3]==0)/len(data_all_550_permutated))
 
plt.title('Bootstrap for accuracy of \nfirst episode unmedicated subgroup of SZ in Dataset 1', fontsize=25, fontweight='bold')
sns.distplot(acc_random, bins=20, kde=True, color='darkturquoise', hist=True, vertical=False)
plt.tick_params(labelsize=20)
plt.plot([acc_true, acc_true], [0, 5],'--', markersize=15, linewidth=3, color='c')
plt.text(acc_true - 0.04, 5, f"True accuracy =  {acc_true:.2f} \nP = {np.sum(acc_random <= acc_true) / len(acc_random)}", fontsize=20)
plt.xlabel('Accuracy', fontsize=25)
plt.ylabel('Density', fontsize=25)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save to PDF format
plt.tight_layout()
plt.subplots_adjust(wspace =0, hspace =0.3)
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Classification_performances_all.pdf')
pdf.savefig()
pdf.close()
plt.show()
