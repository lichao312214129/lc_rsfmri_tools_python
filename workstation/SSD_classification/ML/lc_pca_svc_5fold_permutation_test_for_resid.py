# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a  linear svc model using training dataset, 
and test this model using test dataset with pooling cross-validation stratage.

All datasets (4 datasets) were concatenate into one single dataset, then using cross-validation strategy.

Classfier: Linear SVC
Dimension reduction: PCA

@author: LI Chao
Email: lichao19870617@gmail.com
"""


import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent import futures

from eslearn.feature_engineering.feature_preprocessing import el_preprocessing
from eslearn.feature_engineering.feature_reduction import el_dimreduction
from eslearn.utils.lc_evaluation_model_performances import eval_performance
from eslearn.utils.lc_read_write_mat import read_mat, write_mat

class PCASVCPooling():
    """
    Parameters:
    ----------
        dataset_our_center_550 : path str
            path of dataset 1. 
            NOTE: The first column of the dataset is subject unique index, the second is the diagnosis label(0/1),
            the rest of columns are features. The other dataset are the same as this dataset.

        dataset_206: path str
            path of dataset 2

        dataset_COBRE: path str
            path of dataset 3

        dataset_UCAL: path str
            path of dataset 4

        is_dim_reduction： bool
            if perform dimension reduction (PCA)

        components: float
            How many percentages of the cumulatively explained variance to be retained. This is used to select the top principal components.

        cv: int
            How many folds of the cross-validation.

        out_name: str
            The name of the output results.

    Returns:
    --------
        Classification results, such as accuracy, sensitivity, specificity, AUC and figures that used to report.

    """
    def __init__(self,
                 dataset_our_center_550=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy',
                 dataset_206=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy',
                 dataset_COBRE=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_COBRE.npy',
                 dataset_UCAL=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_UCLA.npy',
                 resid_all=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_site_sex_motion_all.mat',
                 n_perm = 500,
                 is_dim_reduction=True,
                 components=0.95,
                 cv=5):

        self.dataset_our_center_550 = dataset_our_center_550
        self.dataset_206 = dataset_206
        self.dataset_COBRE = dataset_COBRE
        self.dataset_UCAL = dataset_UCAL
        self.resid_all = resid_all
        self.n_perm = n_perm

        self.is_dim_reduction = is_dim_reduction
        self.components = components
        self.cv = cv

    def permutation(self):
        # load data
        data_all = read_mat(self.resid_all)

        # Extracting features and label
        uid_all = data_all[:,0]
        site = data_all[:,2]
        label_all = data_all[:,1]
        feature_all = data_all[:,3:]
        accuracy, sensitivity, specificity, AUC = np.array([]), np.array([]), np.array([]), np.array([])
        print('Permutation testing...\n')
        with ThreadPoolExecutor(4) as executor:   
            to_do = []             
            for i in range(self.n_perm):
                label_all_perm = np.random.permutation(label_all)
                future = executor.submit(self.main_function, label_all_perm, feature_all)
                to_do.append(future)
                
        results = []
        for future in futures.as_completed(to_do):
            res = future.result()
            results.append(np.array(res))
            
        print("Get real performances...\n")
        results_real = self.main_function(label_all, feature_all)  
        results_real = np.array(results_real).reshape(-1, len(results_real))
        results = np.array(results)
        results_all = np.vstack([results_real, results])
        
        np.save(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_real_feu.npy', results_real)
        
        return results_all

    def main_function(self, label_all_perm, feature_all):
        """The training data, validation data and  test data are randomly splited
        """
        print("One permutaion...\n")
        # KFold Cross Validation
        accuracy, sensitivity, specificity, AUC = np.array([]), np.array([]), np.array([]), np.array([])      
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=0)
        for i, (tr_ind, te_ind) in enumerate(kf.split(feature_all)):
            feature_train = feature_all[tr_ind, :]
            label_train = label_all_perm[tr_ind]
            feature_test = feature_all[te_ind, :]
            label_test = label_all_perm[te_ind]

            # normalization
            prep = el_preprocessing.Preprocessing(data_preprocess_method='StandardScaler', data_preprocess_level='group')
            feature_train, feature_test = prep.data_preprocess(feature_train, feature_test)

            # dimension reduction
            if self.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = el_dimreduction.pca_apply(
                    feature_train, feature_test, self.components
                )
                
            # train
            model = self.training(feature_train, label_train)
                
            # test
            pred, dec = self.testing(model, feature_test)

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=0, is_showfig=0)
            accuracy = np.append(accuracy, acc)
            sensitivity = np.append(sensitivity, sens)
            specificity = np.append(specificity, spec)
            AUC = np.append(AUC, auc)

        return np.mean(accuracy),np.mean(sensitivity), np.mean(specificity), np.mean(AUC)

    def dimReduction(self, train_X, test_X, pca_n_component):
        train_X, trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def training(self, train_X, train_y):  
        svc = svm.LinearSVC(class_weight='balanced', random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def testing(self, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(self, data, name):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump(data, f)
            
    def save_fig(self, out_name):
        # Save ROC and Classification 2D figure
        (acc, sens, spec, auc) = eval_performance(
            label_all, self.prediction, self.decision, 
            self.accuracy, self.sensitivity, self.specificity, self.AUC,
            verbose=0, is_showfig=1, legend1='HC', legend2='SZ', is_savefig=1, 
            out_name=out_name
        )
#
if __name__ == '__main__':
    clf=PCASVCPooling()
    results=clf.permutation()
    clf.save_results(results, r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\permutation_feu.npy')
    print("Done!")
    
    dd = np.load( r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\permutation_feu.npy', allow_pickle=True)
    
    # resid_all = read_mat(clf.resid_all)
    
    # feu = r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_firstepisode_and_unmedicated_550.npy'
    # feu = np.load(feu)
    # resid_feu = resid_all[np.in1d(resid_all[:,0],feu[:,0])]
    # write_mat(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\resid_firstepisode_and_unmedicated_550.mat', 'resid_firstepisode_and_unmedicated_550', resid_feu)
    
    
    
