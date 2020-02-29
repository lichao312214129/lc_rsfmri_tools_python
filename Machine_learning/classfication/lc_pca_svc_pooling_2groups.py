# -*- coding: utf-8 -*-
"""
Created on 2020/02/08
Classifier: linear SVC
Dimension reduction: PCA
Feature selection: Relief-based feature selection algorithm.
------
@author: LI Chao
Upgrade: Added feature selection using Relief-based feature selection algorithms (2020/02/22).
"""
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
import nibabel as nib
import os

from Utils.lc_evaluation_model_performances import eval_performance
from Utils.lc_niiProcessor import NiiProcessor


class PCASVCPooling():
    """
    This class is used to execute pca-svc-based classification training and testing.
    NOTE: Input data must be in the .nii or similar format.
    TODO: Muticlass classification.

    Parameters
    ----------
    path_patients :    
        Path of the image files of patients

    path_HC : 
        Path of the image files of HC 

    path_mask : 
        Path of the mask

    path_out : 
        Path to save results
        
    is_dim_reduction : bool
        If perfrome dimension reduction.

    is_feature_selection : bool
        if perfrome feature selection.

    n_features_to_select: int
        number of features to select

    n_components: float from 0 to 1
        If is_dim_reduction, then how many components to remain.

    num_of_kfold: int
        Number of the k in k-fold cross-validation

    is_showfig_finally: bool
        If show figure after all iteration finished.

    is_showfig_in_each_fold: bool
        If show figure in each fold.

    Returns
    -------
    Save all classification results and figures to local disk.
    """
    def __init__(self,
                 path_patients=None,
                 path_HC=None,
                 path_mask=None,
                 path_out=None,
                 num_of_kfold=5,
                 is_dim_reduction=1,
                 components=0.95,
                 is_feature_selection=False,
                 n_features_to_select=None,
                 is_showfig_finally=True,
                 is_showfig_in_each_fold=False):

         self.path_patients = path_patients
         self.path_HC = path_HC
         self.path_mask = path_mask
         self.path_out = path_out
         self.num_of_kfold =  num_of_kfold
         self.is_dim_reduction = is_dim_reduction
         self.components = components
         self.is_feature_selection = is_feature_selection
         self.n_features_to_select = n_features_to_select
         self.is_showfig_finally = is_showfig_finally
         self.is_showfig_in_each_fold = is_showfig_in_each_fold


    def main_function(self):
        """
        """
        print('Training model and testing...\n')

        # load data and mask
        data_all, label_all, self.orig_shape, self.mask_obj, self.mask_all = self._load_nii_and_gen_label(self.path_mask)

        # KFold Cross Validation
        self.label_test_all = np.array([], dtype=np.int16)
        train_index = np.array([], dtype=np.int16)
        test_index = np.array([], dtype=np.int16)
        self.decision = np.array([], dtype=np.int16)
        self.prediction = np.array([], dtype=np.int16)
        self.accuracy = np.array([], dtype=np.float16)
        self.sensitivity = np.array([], dtype=np.float16)
        self.specificity = np.array([], dtype=np.float16)
        self.AUC = np.array([], dtype=np.float16)
        self.coef = []
        kf = KFold(n_splits=self.num_of_kfold, shuffle=True, random_state=0)
        for i, (tr_ind, te_ind) in enumerate(kf.split(data_all)):
            print(f'------{i+1}/{self.num_of_kfold}...------\n')
            train_index = np.int16(np.append(train_index, tr_ind))
            test_index = np.int16(np.append(test_index, te_ind))
            feature_train = data_all[tr_ind, :]
            label_train = label_all[tr_ind]
            feature_test = data_all[te_ind, :]
            label_test = label_all[te_ind]
            self.label_test_all = np.int16(np.append(self.label_test_all, label_test))

            # Resampling training data
            feature_train, label_train = self.re_sampling(
                feature_train, label_train)

            # Normalization
            feature_train = self.normalization(feature_train)
            feature_test = self.normalization(feature_test)

            # dimension reduction using univariate feature selection
            # feature_train, feature_test, mask_selected = self.dimReduction_filter(
            #         feature_train, label_train, feature_test, 0.05)

            # Dimension reduction using PCA
            if self.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = self.dimReduction_PCA(
                    feature_train, feature_test, self.components)
                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')
                print(f'The feature number is {feature_train.shape[1]}')
                
            # Feature selection
            if self.is_feature_selection:
                feature_train, feature_test, mask, n_features_origin = self.feature_selection_relief(feature_train, 
                                                                            label_train, 
                                                                            feature_test, 
                                                                            self.n_features_to_select)       
            # Train and test
            print('training and testing...\n')
            model = self.training(feature_train, label_train)
            
            # Get weight 
            if self.is_feature_selection:
                coef = np.zeros([n_features_origin,])
                coef[mask] = model.coef_
            else:
                coef = model.coef_
                
            if self.is_dim_reduction:
                self.coef.append(model_dim_reduction.inverse_transform(coef))
            else:
                self.coef.append(coef)

            pred, dec = self.testing(model, feature_test)
            self.prediction = np.append(self.prediction, np.array(pred))
            self.decision = np.append(self.decision, np.array(dec))

            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=1, is_showfig=self.is_showfig_in_each_fold)

            self.accuracy = np.append(self.accuracy, acc)
            self.sensitivity = np.append(self.sensitivity, sens)
            self.specificity = np.append(self.specificity, spec)
            self.AUC = np.append(self.AUC, auc)
            
        # Save results and fig to local path
        self.save_results()
        self._weight2nii(dimension_nii_data=(61, 73, 61))
        self.save_fig()
            
        print("--" * 10 + "Done!" + "--" * 10 )
        return self


    def _load_nii_and_gen_label(self, path_mask):
            """
            Load nii and generate label
            """
            data1, _ = NiiProcessor().main(self.path_patients)
            data1 = np.squeeze(
                np.array([np.array(data1).reshape(1, -1) for data1 in data1]))
    
            data2, _ = NiiProcessor().main(self.path_HC)
            data2 = np.squeeze(
                np.array([np.array(data2).reshape(1, -1) for data2 in data2]))
    
            data = np.vstack([data1, data2])
    
            # data in mask
            mask, mask_obj = NiiProcessor().read_sigle_nii(path_mask)
            orig_shape = mask.shape
            mask = mask >= 0.2
            mask = np.array(mask).reshape(-1,)

            data_in_mask = data[:, mask]
            # label
            label = np.hstack(
                [np.ones([len(data1), ]), np.ones([len(data2), ])-1])
            return data_in_mask, label, orig_shape, mask_obj, mask
    

    def re_sampling(self, feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(f"After re-sampling, the sample size are: {sorted(Counter(label_resampled).items())}")
        return feature_resampled, label_resampled

    def normalization(self, data):
        '''
        Because of our normalization level is on subject, 
        we should transpose the data matrix on python(but not on matlab)
        '''
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data = scaler.transform(data.T) .T
        return z_data
    
    def feature_selection_relief(self, feature_train, label_train, feature_test, n_features_to_select=None):
        """
        This functio is used to select the features using relief-based feature selection algorithms
        """
        from skrebate import ReliefF
        
        [n_sub, n_features] = np.shape(feature_train)
        if n_features_to_select is None: 
            n_features_to_select = np.int(np.round(n_features / 10))
            
        if isinstance(n_features_to_select, np.float): 
            n_features_to_select = np.int(np.round(n_features * n_features_to_select))
        
        fs = ReliefF(n_features_to_select=n_features_to_select, 
                     n_neighbors=100, discrete_threshold=10, verbose=True, n_jobs=-1)
        fs.fit(feature_train, label_train)
        feature_train = fs.transform(feature_train)
        feature_test = fs.transform(feature_test)
        mask = fs.top_features_[:n_features_to_select]
        return feature_train, feature_test, mask, n_features
        
        
    def dimReduction_filter(self, feature_train, label_train, feature_test, p_thrd = 0.05):
        """
        This function is used to Univariate Feature Selection:: ANOVA
        """
        from sklearn.feature_selection import f_classif
        f, p = f_classif(feature_train, label_train)
        mask_selected = p < p_thrd
        feature_train = feature_train[:,mask_selected]
        feature_test = feature_test[:, mask_selected]
        return feature_train, feature_test, mask_selected
        
    def dimReduction_PCA(self, train_X, test_X, pca_n_component):
        from Utils.lc_dimreduction import pca_apply
        x_train, x_test, trained_pca = pca_apply(
                    train_X, test_X, pca_n_component)
        return x_train, x_test, trained_pca

    def training(self, train_X, train_y):
        # Classfier is SVC
        svc = svm.SVC(kernel='linear', C=1, class_weight='balanced', max_iter=5000, random_state=0)
        svc.fit(train_X, train_y)
        return svc

    def testing(self, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(self):
        # Save performances and others
        import pandas as pd
        performances_to_save = np.concatenate([[self.accuracy], [self.sensitivity], [self.specificity], [self.AUC]], axis=0).T
        de_pred_label_to_save = np.concatenate([[self.decision], [self.prediction], [self.label_test_all]], axis=0).T

        performances_to_save = pd.DataFrame(performances_to_save, columns=[['Accuracy','Sensitivity', 'Specificity', 'AUC']])
        de_pred_label_to_save = pd.DataFrame(de_pred_label_to_save, columns=[['Decision','Prediction', 'Sorted_Real_Label']])
        
        performances_to_save.to_csv(os.path.join(self.path_out, 'Performances.txt'), index=False, header=True)
        de_pred_label_to_save.to_csv(os.path.join(self.path_out, 'Decision_prediction_label.txt'), index=False, header=True)

        
    def _weight2nii(self, dimension_nii_data=(61, 73, 61)):
        """
        Transfer weight matrix to nii file
        I used the mask file as reference to generate the nii file
        """
        weight = np.squeeze(self.coef)
        weight_mean = np.mean(weight, axis=0)

        # to orignal space
        weight_mean_orig = np.zeros(np.size(self.mask_all))
        weight_mean_orig[self.mask_all] = weight_mean
        weight_mean_orig =  np.reshape(weight_mean_orig, dimension_nii_data)
        # save to nii
        weight_nii = nib.Nifti1Image(weight_mean_orig, affine=self.mask_obj.affine)
        weight_nii.to_filename(os.path.join(self.path_out, 'weight.nii'))
        
    def save_fig(self):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(self.label_test_all, self.prediction, self.decision, 
                                                self.accuracy, self.sensitivity, self.specificity, self.AUC,
                                                verbose=0, is_showfig=self.is_showfig_finally, is_savefig=1, 
                                                out_name=os.path.join(self.path_out, 'Classification_performances.pdf'))

#
if __name__ == '__main__':
    # =============================================================================
    # All inputs
    path_patients = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_patient\P_Weighted_selected'
    path_HC = r'D:\WorkStation_2018\Workstation_Old\WorkStation_2018-05_MVPA_insomnia_FCS\Degree\degree_gray_matter\Zdegree\Z_degree_control\C_Weighted_selected'
    path_mask = r'G:\Softer_DataProcessing\spm12\spm12\tpm\Reslice3_TPM_greaterThan0.2.nii'
    path_out = r'D:\workstation_b\haoge\FC'
    # =============================================================================
    
    clf = PCASVCPooling(path_patients=path_patients,
                        path_HC=path_HC,
                        path_mask=path_mask,
                        path_out=path_out,
                        is_feature_selection=True, 
                        n_features_to_select=0.99,
                        components=0.75)

    results = clf.main_function()
    results = results.__dict__
    
    print(f"mean accuracy = {np.mean(results['accuracy'])}")
    print(f"std of accuracy = {np.std(results['accuracy'])}")

    print(f"mean sensitivity = {np.mean(results['sensitivity'])}")
    print(f"std of sensitivity = {np.std(results['sensitivity'])}")

    print(f"mean specificity = {np.mean(results['specificity'])}")
    print(f"std of specificity = {np.std(results['specificity'])}")

    print(f"mean AUC = {np.mean(results['AUC'])}")
    print(f"std of AUC = {np.std(results['AUC'])}")
