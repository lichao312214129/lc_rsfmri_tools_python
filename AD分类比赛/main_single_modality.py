# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:45:42 2020

@author: Li Chao, Dong Mengshi
"""

import numpy as np
import pickle
import os

from model import Model
from split import data_train, data_validation, data_test, label_train, label_validation, label_test

def main(metric="FA", include_diagnoses=(1,3)):
    """Training and validation

    Parameters:
    ----------
    metric: string
        metric name, such as "FA" or "MD"

    include_diagnoses: tuple or list
        which groups/diagnoses included in training and validation, such as (1,3)
    """

    # Instantiate model
    model = Model()
    
    # Get NC and AD
    label_train_ =  label_train[np.in1d(label_train, include_diagnoses)]
    data_train_ = data_train[metric][np.in1d(label_train, include_diagnoses)]
    label_validation_ =  label_validation[np.in1d(label_validation, include_diagnoses)]
    data_validation_ = data_validation[metric][np.in1d(label_validation, include_diagnoses)]
    
    # Denan
    data_train_, label_train_ = model.denan(data_train_, label_train_)
    data_validation_, label_validation_ = model.denan(data_validation_, label_validation_)
    
    # # Preprocessing
    scaler, data_train_ = model.preprocess_(data_train_)
    data_validation_ = scaler.transform(data_validation_)
    
    # Feature selection
    # rfecv, data_train_ = model.feature_selection(LinearSVC(random_state=666), data_train_, label_train_)
    # data_validation_ = rfecv.transform(data_validation_)
        
    # Fit
    clf1 = model.train_linearSVC(data_train_, label_train_)
    clf2 = model.train_logistic_regression(data_train_, label_train_)
    clf3 = model.train_logistic_regression(data_train_, label_train_)
    clf4 = model.train_ridge_regression(data_train_, label_train_)
    clf5 = model.train_randomforest(data_train_, label_train_)
    clfs = [clf1, clf4]
    
    # Merge models
    merged_model = model.merge_models(data_train_, label_train_, *clfs)
    
    # Dict models and scaler
    all_models = {"orignal_models": clfs, "merged_model": merged_model, "scaler": scaler}
    
    # Save original models, merged model and scaler
    groups = ["nc","mci","ad"]
    save_name = [groups[include_diagnoses_-1] for include_diagnoses_ in include_diagnoses]
    save_name = "model_" + "VS".join(save_name) + ".pickle.dat"
    save_file = os.path.join("D:\My_Codes\lc_private_codes\AD分类比赛", save_name)
    pickle.dump(all_models, open(save_file, "wb"))
    
    # Predict
    predict_proba_train, prediction_train = model.merge_predict(all_models["merged_model"], data_train_, *all_models["orignal_models"])
    predict_proba_validation, prediction_validation = model.merge_predict(all_models["merged_model"], data_validation_, *all_models["orignal_models"])
    
    # predict_proba_train, prediction_train = model.predict(clf1, data_train_)
    # predict_proba_validation, prediction_validation = model.predict(clf1, data_validation_)
    
    # Evaluation
    acc_train, auc_train, f1_train, confmat_train = model.evaluate(label_train_, predict_proba_train, prediction_train)
    acc_validation, auc_validation, f1_validation, confmat_validation = model.evaluate(label_validation_, predict_proba_validation, prediction_validation)
    print(f"Traing dataset:\nacc = {acc_train}\nauc = {auc_train}\nf1score = {f1_train}\n")
    print(f"Test dataset:\nacc = {acc_validation}\nauc = {auc_validation}\nf1score = {f1_validation}\n")
    # print(f"Model is {clf1.best_estimator_}")


if __name__ ==  "__main__":
    main("FA")