# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a linear svc model using training dataset, 
and test this model using test dataset with leave-one site cross-validation stratage.
Classfier: Linear SVC
Dimension reduction: PCA

@author: LI Chao
Email: lichao19870617@gmail.com
"""

import sys
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from eslearn.feature_engineering.feature_preprocessing import el_preprocessing
from eslearn.feature_engineering.feature_reduction import el_dimreduction
from eslearn.utils.lc_evaluation_model_performances import eval_performance


class SVCRFECV():
    """
    Parameters:
    ----------
        dataset_our_center_550 : path str
            path of dataset 1
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
    def __init__(
         sel,
         dataset_our_center_550=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_550.npy',
         dataset_206=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy',
         dataset_COBRE=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_COBRE.npy',
         dataset_UCAL=r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_UCLA.npy',
         is_dim_reduction=True,
         components=0.95,
         cv=5
    ):

        sel.dataset_our_center_550 = dataset_our_center_550
        sel.dataset_206 = dataset_206
        sel.data_COBRE = dataset_COBRE
        sel.data_UCAL = dataset_UCAL

        sel.is_dim_reduction = is_dim_reduction
        sel.components = components
        sel.cv = cv

    def main_svc_rfe_cv(sel):
        print('Training model and testing...\n')
        # Load data
        uid_550, feature_550, label_550 = sel._load_data(sel.dataset_our_center_550)
        uid_206, feature_206, label_206 = sel._load_data(sel.dataset_206)
        uid_COBRE, feature_COBRE, label_COBRE = sel._load_data(sel.data_COBRE)
        uid_UCAL, feature_UCAL, label_UCAL = sel._load_data(sel.data_UCAL)
        uid_all = np.concatenate([uid_550, uid_206, uid_COBRE, uid_UCAL])
        feature_all = [feature_550, feature_206, feature_COBRE, feature_UCAL]
        sel.label_all = [label_550, label_206, label_COBRE, label_UCAL]
        name = ['550','206','COBRE','UCLA']

        # Leave one site CV
        n_site = len(sel.label_all)
        test_index = np.array([], dtype=np.int16)
        sel.decision = np.array([], dtype=np.int16)
        sel.prediction = np.array([], dtype=np.int16)
        sel.accuracy = np.array([], dtype=np.float16)
        sel.sensitivity = np.array([], dtype=np.float16)
        sel.specificity = np.array([], dtype=np.float16)
        sel.AUC = np.array([], dtype=np.float16)
        sel.coef = []
        for i in range(n_site):
            print('-'*40)
            print(f'{i+1}/{n_site}: test dataset is {name[i]}...')
            feature_train, label_train = feature_all.copy(), sel.label_all.copy()
            feature_test, label_test = feature_train.pop(i), label_train.pop(i)
            feature_train = np.concatenate(feature_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)

            # Resampling training data
            # feature_train, label_train = sel.re_sampling(feature_train, label_train)
            # Normalization
            prep = el_preprocessing.Preprocessing(data_preprocess_method='StandardScaler', data_preprocess_level='subject')
            feature_train, feature_test = prep.data_preprocess(feature_train, feature_test)

            # Dimension reduction
            if sel.is_dim_reduction:
                feature_train, feature_test, model_dim_reduction = el_dimreduction.pca_apply(
                    feature_train, feature_test, sel.components
                )

                print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
            else:
                print('No dimension reduction perfromed\n')

            # Train and test
            print('training and testing...\n')
            model = sel.training(feature_train, label_train, sel.cv)
        
            if sel.is_dim_reduction:
                sel.coef.append(model_dim_reduction.inverse_transform(model.coef_))  # save coef
            else:
                sel.coef.append(clf.coef_)  # save coef

            pred, dec = sel.testing(model, feature_test)
            sel.prediction = np.append(sel.prediction, np.array(pred))
            sel.decision = np.append(sel.decision, np.array(dec))
            
            # Evaluating classification performances
            acc, sens, spec, auc = eval_performance(label_test, pred, dec, 
                accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
                 verbose=1, is_showfig=0)
            sel.accuracy = np.append(sel.accuracy, acc)
            sel.sensitivity = np.append(sel.sensitivity, sens)
            sel.specificity = np.append(sel.specificity, spec)
            sel.AUC = np.append(sel.AUC, auc)
            print(f'performances = {acc, sens, spec,auc}')
        
        # sel.label_all = np.concatenate(sel.label_all)
        sel.special_result = np.concatenate( [uid_all, sel.label_all, sel.decision, sel.prediction], axis=0).reshape(4, -1).T
        return sel

    def _load_data(sel, data_path):
        data = np.load(data_path)
        name = data_path.split('\\')[-1]
        print(f"Dataset is {name}")
        uid = data[:,0]
        feature = data[:, 2:]
        label = data[:, 1]
        return uid, feature, label

    def re_sampling(sel, feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        from collections import Counter
        print(sorted(Counter(label_resampled).items()))
        return feature_resampled, label_resampled

    def dimReduction(sel, train_X, test_X, pca_n_component):
        train_X, trained_pca = dimreduction.pca(train_X, pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def training(sel, train_X, train_y, cv):
        #
        model =  LogisticRegression(class_weight='balanced')
        model.fit(train_X, train_y)
        return model

    def testing(sel, model, test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict, decision

    def save_results(sel, data, name):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump(data, f)

    def save_fig(sel, out_name):
        # Save ROC and Classification 2D figure
        acc, sens, spec, auc = eval_performance(sel.label_all, sel.prediction, sel.decision, 
                                                sel.accuracy, sel.sensitivity, sel.specificity, sel.AUC,
                                                verbose=0, is_showfig=1, legend1='HC', legend2='SSD', is_savefig=1, 
                                                out_name=out_name)
#
if __name__ == '__main__':
    clf = SVCRFECV()
    results = clf.main_svc_rfe_cv()

    # clf.save_fig(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\performances_svc_leave_one_site_cv.pdf')
    
    results = results.__dict__
    clf.save_results(results, r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\results_svc_leave_one_site_cv.npy')

    # print(np.mean(clf.accuracy))
    # print(np.std(clf.accuracy))

    # print(np.mean(clf.sensitivity))
    # print(np.std(clf.sensitivity))

    # print(np.mean(clf.specificity))
    # print(np.std(clf.specificity))
    
    # print(np.mean(clf.AUC))
    # print(np.std(clf.AUC))
