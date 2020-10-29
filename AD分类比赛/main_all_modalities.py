# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:45:42 2020

@author: Li Chao, Dong Mengshi
"""

import numpy as np
import pickle
import os

from model import Model
from split import (data_train, data_validation, data_test, 
    label_train, label_validation, label_test, 
    mean_num_na_train,mean_num_na_validation,mean_num_na_test, 
    demo_train, demo_validation, demo_test)
from preprocess import METRICS


def main(include_diagnoses=(1,3)):
    """Training and validation using all modalities

    Parameters:
    ----------
    include_diagnoses: tuple or list
        which groups/diagnoses included in training and validation, such as (1,3)
    """

    # Instantiate model
    model = Model()
    
    # Concatenate all modalities, demo and mean_num_na
    for i, metrics_ in enumerate(METRICS):
        if i == 0:
            data_train_all_mod = data_train[metrics_]
            data_validation_all_mod = data_validation[metrics_]
            data_test_all_mod = data_test[metrics_]
        else:
            data_train_all_mod = np.hstack([data_train_all_mod, data_train[metrics_]])
            data_validation_all_mod = np.hstack([data_validation_all_mod, data_validation[metrics_]])
            data_test_all_mod = np.hstack([data_test_all_mod, data_test[metrics_]])
            
    # Get included diagnoses
    idx_train = np.in1d(label_train, include_diagnoses)
    idx_validation = np.in1d(label_validation, include_diagnoses)
    idx_test = np.in1d(label_test, include_diagnoses)

    label_train_ = label_train[idx_train]
    data_train_ = data_train_all_mod[idx_train]
    # demo_train_ = demo_train[idx_train]
    # mean_num_na_train_ = mean_num_na_train[idx_train].reshape([-1,1])

    label_validation_ =  label_validation[idx_validation]
    data_validation_ = data_validation_all_mod[idx_validation]
    # demo_validation_ = demo_validation[idx_validation]
    # mean_num_na_validation_ = mean_num_na_validation[idx_validation].reshape([-1,1])

    label_test_ =  label_test[idx_test]
    data_test_ = data_test_all_mod[idx_test]
    # demo_test_ = demo_test[idx_test]
    # mean_num_na_test_ = mean_num_na_test[idx_test].reshape([-1,1])

    # Concatenate all features
    # data_train_ = np.hstack([data_train_, mean_num_na_train_, demo_train_])
    # data_validation_ = np.hstack([data_validation_, mean_num_na_validation_, demo_validation_])

    # Denan
    data_train_, label_train_ = model.denan(data_train_, label_train_)
    data_validation_, label_validation_ = model.denan(data_validation_, label_validation_)
    data_test_, label_test_ = model.denan(data_test_, label_test_)
    
    # Preprocessing
    scaler, data_train_ = model.preprocess_(data_train_)
    data_validation_ = scaler.transform(data_validation_)
    data_test_ = scaler.transform(data_test_)
    
    # Feature selection
    # rfecv, data_train_ = model.feature_selection(LinearSVC(random_state=666), data_train_, label_train_)
    # data_validation_ = rfecv.transform(data_validation_)
        
    # Fit
    clf1 = model.train_linearSVC(data_train_, label_train_)
    clf2 = model.train_SVC(data_train_, label_train_)
    # clf3 = model.train_logistic_regression(data_train_, label_train_)
    # clf4 = model.train_ridge_regression(data_train_, label_train_)
    # clf5 = model.train_randomforest(data_train_, label_train_)
    clfs = [clf1, clf2]
    
    # Merge models
    merged_model = model.merge_models(data_train_, label_train_, *clfs)
    
    # Dict models and scaler
    all_models = {"orignal_models": clfs, "merged_model": merged_model, "scaler": scaler}
    
    # Save original models, merged model and scaler
    groups = ["nc","mci","ad"]
    save_name = [groups[include_diagnoses_-1] for include_diagnoses_ in include_diagnoses]
    save_name = "model_all_modalities_" + "VS".join(save_name) + ".pickle.dat"
    save_file = os.path.join("D:\My_Codes\lc_private_codes\AD分类比赛", save_name)
    pickle.dump(all_models, open(save_file, "wb"))
    
    # Predict
    predict_proba_train, prediction_train = model.merge_predict(all_models["merged_model"], data_train_, *all_models["orignal_models"])
    predict_proba_validation, prediction_validation = model.merge_predict(all_models["merged_model"], data_validation_, *all_models["orignal_models"])
    predict_proba_test, prediction_test = model.merge_predict(all_models["merged_model"], data_test_, *all_models["orignal_models"])
    
    # predict_proba_train, prediction_train = model.predict(clf2, data_train_)
    # predict_proba_validation, prediction_validation = model.predict(clf2, data_validation_)
    
    # Evaluation
    acc_train, auc_train, f1_train, confmat_train = model.evaluate(label_train_, predict_proba_train, prediction_train)
    acc_validation, auc_validation, f1_validation, confmat_validation = model.evaluate(label_validation_, predict_proba_validation, prediction_validation)
    acc_test, auc_test, f1_test, confmat_test = model.evaluate(label_test_, predict_proba_test, prediction_test)
    print(f"Traing dataset:\nacc = {acc_train}\nauc = {auc_train}\nf1score = {f1_train}\n")
    print(f"Validation dataset:\nacc = {acc_validation}\nauc = {auc_validation}\nf1score = {f1_validation}\n")
    # print(f"Test dataset:\nacc = {acc_test}\nauc = {auc_test}\nf1score = {f1_test}\n")
    # print(f"Model is {clf1.best_estimator_}")
    return (predict_proba_train, prediction_train, predict_proba_validation, prediction_validation)


if __name__ ==  "__main__":
    predict_proba_train, prediction_train, predict_proba_validation, prediction_validation = main(include_diagnoses=(1,3))