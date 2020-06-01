import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

permutation_pooledcv = np.load(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\permutation_pooledcv.npy', allow_pickle=True)
permutation_feu = np.load(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\permutation_feu.npy', allow_pickle=True)

# Pooled
plt.figure(figsize=(12,7))
plt.subplot(3,3,1)
plt.hist(permutation_pooledcv[1:,0], color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[0,0],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_pooledcv[1:,0] > permutation_pooledcv[0,0])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Accuracy\np={pvalue:.3f}')

plt.subplot(3,3,4)
plt.hist(permutation_pooledcv[1:,1], color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[0,1],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_pooledcv[1:,1] > permutation_pooledcv[0,1])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Sensitivity\np={pvalue:.3f}')

plt.subplot(3,3,7)
plt.hist(permutation_pooledcv[1:,2], color='darkturquoise', alpha=0.9)
plt.plot(permutation_pooledcv[0,2],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_pooledcv[1:,2] > permutation_pooledcv[0,2])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Specificity\np={pvalue:.3f}')

# FEU
plt.subplot(3,3,2)
plt.hist(permutation_feu[1:,0], color='paleturquoise', alpha=0.9)
plt.plot(permutation_feu[0,0],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_feu[1:,0] > permutation_feu[0,0])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Accuracy\np={pvalue:.3f}')

plt.subplot(3,3,5)
plt.hist(permutation_feu[1:,1], color='paleturquoise', alpha=0.9)
plt.plot(permutation_feu[0,1],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_feu[1:,1] > permutation_feu[0,1])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Sensitivity\np={pvalue:.3f}')

plt.subplot(3,3,8)
plt.hist(permutation_feu[1:,2], color='paleturquoise', alpha=0.9)
plt.plot(permutation_feu[0,2],7, '*', markersize=15, color='orange')
pvalue = (np.sum(permutation_feu[1:,2] > permutation_feu[0,2])+1)/(500+1)
plt.ylabel('Frequency')
plt.title(f'Specificity\np={pvalue:.3f}')

#%% performances
# Inputs
classification_results_pooling_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_real_pooled.npy'
classification_results_results_leave_one_site_cv_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_fc_excluded_greater_fd_and_regressed_out_site_sex_motion_all.npy'
classification_results_feu_file = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_real_feu.npy'

# Load and proprocess
results_pooling = np.load(classification_results_pooling_file, allow_pickle=True)
results_leave_one_site_cv = np.load(classification_results_results_leave_one_site_cv_file, allow_pickle=True)
results_feu = np.load(classification_results_feu_file, allow_pickle=True)

#
accuracy_pooling, sensitivity_pooling, specificity_pooling = results_pooling[0,:], results_pooling[1,:], results_pooling[2,:]
performances_pooling = [accuracy_pooling, sensitivity_pooling, specificity_pooling]
performances_pooling = pd.DataFrame(performances_pooling)

accuracy_feu, sensitivity_feu, specificity_feu = results_feu[0,:], results_feu[1,:], results_feu[2,:]
performances_feu = [accuracy_feu, sensitivity_feu, specificity_feu]
performances_feu = pd.DataFrame(performances_feu)

# Bar: performances in the whole Dataset.
plt.subplot(1,3,3)
all_mean = np.concatenate([np.mean(performances_pooling.values,1), np.mean(performances_feu.values,1)])
error = np.concatenate([np.std(performances_pooling.values, 1), np.std(performances_feu.values, 1)])

color = ['darkturquoise'] * 3 +  ['paleturquoise'] * 3
plt.bar(np.arange(0,len(all_mean)), all_mean, yerr = error, 
        capsize=5, linewidth=2, color=color)
# plt.tick_params(labelsize=10)
plt.xticks(np.arange(0,len(all_mean)), ['Accuracy', 'Sensitivity', 'Sensitivity'] * 3, fontsize=10, rotation=45, ha='right')
plt.title('Classification performances')
y_major_locator=MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.grid(axis='y')
# plt.fill_between(np.linspace(-0.4,2.4), 1.01, 1.08, color='darkturquoise', alpha=0.9)
# plt.fill_between(np.linspace(2.6, 5.4), 1.01, 1.08, color='paleturquoise', alpha=0.9)

    
plt.subplots_adjust(wspace=0.4, hspace=0.5)
# plt.tight_layout()
pdf = PdfPages(r'D:\WorkStation_2018\SZ_classification\Figure\Processed\permutation_test.pdf')
pdf.savefig()
pdf.close()
plt.show()
print('-'*50)
