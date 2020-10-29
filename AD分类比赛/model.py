# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:58:29 2020

@author: Li Chao, Dong Mengshi
"""

import pandas  as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import metrics

from split import data_train, data_validation, data_test, label_train, label_validation, label_test


class Model(object):

    def __init__(self):
        self.file = r'F:\AD分类比赛\MCAD_AFQ_competition.mat'
        self.seed = 666
        # self.pca_n_component = np.linspace(0.5, 0.99, 2)
        self.pca_n_component = [0.95]
        # self.regularization_strength = np.linspace(0.0001, 100, 2)
        self.regularization_strength = [0.0001]
    
    def denan(self, data, label):
        """
        De-nan 
        """

        idx_nan = np.sum(np.isnan(data),axis=1) > 0  # Drop the case if have any nan
        if not isinstance(data, pd.core.frame.DataFrame):
            data = pd.DataFrame(data)
        data_ = data.dropna().values
        label_ = label[idx_nan == False]
        return data_, label_
    
    def preprocess_(self, data):
        """
        Preprocess
        """

        scaler = StandardScaler()
        data_ = scaler.fit_transform(data)
        return scaler, data_

    def _preprocess_for_all(self, data, label):
        data_ = {}
        label_ = {}
        for key in data.keys():
            idx_nan = np.sum(np.isnan(data[key]),axis=1) > 0  # Drop the case if have any nan
            data_[key] = pd.DataFrame(data[key]).dropna()
            label_[key] = label[idx_nan == False]
        return data_, label_
    
    def feature_selection(self, model, data, label):
        rfecv = RFECV(estimator=model, step=.1, cv=StratifiedKFold(3), scoring='accuracy')
        rfecv.fit(data, label)
        data_  = rfecv.transform(data)
        return rfecv, data_

    def train_ridge_regression(self, feature, label):
        pipe = Pipeline([
        ('reduce_dim', PCA(random_state=self.seed)),
        ('classify', RidgeClassifier(random_state=self.seed)),
        ])

        param_grid = {
            'reduce_dim__n_components': [0.99],
            'classify__alpha': [0.0001]
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)

        grid.fit(feature, label)
        return grid

    def train_logistic_regression(self, feature, label):
        pipe = Pipeline([
        ('reduce_dim', PCA(random_state=self.seed)),
        ('classify', LogisticRegression(random_state=self.seed)),
        ])

        param_grid = {
            'reduce_dim__n_components': self.pca_n_component,
            'classify__C': self.regularization_strength
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)

        grid.fit(feature, label)
        return grid

    def train_linearSVC(self, feature, label):
        pipe = Pipeline([
        ('reduce_dim', PCA(random_state=self.seed)),
        ('classify', LinearSVC(random_state=self.seed)),
        ])

        param_grid = {
            'reduce_dim__n_components': [0.99],
            'classify__C': [0.0001]
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)

        grid.fit(feature, label)
        return grid
    
    def train_SVC(self, feature, label):
        pipe = Pipeline([
        ('reduce_dim', PCA(random_state=self.seed)),
        ('classify', SVC(random_state=self.seed, kernel='rbf')),
        ])

        param_grid = {
            'reduce_dim__n_components': [0.95],
            'classify__C': [2]
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)

        grid.fit(feature, label)
        return grid

    def train_randomforest(self, feature, label):
        pipe = Pipeline([
        ('reduce_dim', PCA(random_state=self.seed)),
        ('classify', RandomForestClassifier(n_estimators=100, random_state=self.seed)),
        ])

        param_grid = {
            'reduce_dim__n_components': self.pca_n_component,
            'classify__n_estimators': [100]
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)

        grid.fit(feature, label)
        return grid

    def merge_models(self, feature, label, *args):
        """Merge all selected models' predictions

        We will train a model with all models' predicted probability as features 
        Parameters:
        ----------
        feature: ~
        label: ~
        *args: original models

        Returns:
        --------
        clf: merged model
        """
        
        if len(args) == 1:
            clf = args[0]
        else:
            predict_results = [self.predict(args_, feature) for args_ in args]
            decision = [pred[0] for pred in predict_results]
            decision = [dec[:,-1] if len(np.shape(dec)) > 1 else dec for dec in decision]
            decision = pd.DataFrame(decision).T.values
            
            # Fit merge-model
            clf = LogisticRegressionCV(cv=10)
            clf.fit(decision, label)
        return clf
    
    def merge_predict(self, clf, feature, *args):
        """Prediction using merged model

        Parameters:
        ----------
        clf: merged model
        feature: ~
        *args: original models
        """
        
        if len(args) == 1:
            clf = args[0]
            # Predict
            predict_proba, prediction = self.predict(clf, feature)
        else:
            predict_results = [self.predict(args_, feature) for args_ in args]
            decision = [pred[0] for pred in predict_results]
            decision = [dec[:,-1] if len(np.shape(dec)) > 1 else dec for dec in decision]
            decision = pd.DataFrame(decision).T.values
            # Predict
            predict_proba, prediction = self.predict(clf, decision)
        return predict_proba, prediction

    def vote_predict(self, clf, feature, *args):
        """Prediction using by vote each model predict

        Parameters:
        ----------
        clf: merged model
        feature: ~
        *args: original models
        """

        predict_results = [self.predict(args_, feature) for args_ in args]
        predict_proba = [pred[0] for pred in predict_results]
        predict_proba = [dec[:,-1] if len(np.shape(dec)) > 1 else dec for dec in predict_proba]
        predict_proba = pd.DataFrame(predict_proba).T.values
        predict_proba = np.mean(predict_proba, axis=1)

        prediction = [pred[1] for pred in predict_results]
        prediction = pd.DataFrame(prediction).T.values
        prediction = np.array([np.argmax(np.bincount(prediction[i,:])) for i in range(len(prediction))])
    
        return predict_proba, prediction
    
    def predict(self, clf, feature):
        prediction = clf.predict(feature)
        
        if hasattr(clf, 'predict_proba'):
            predict_proba = clf.predict_proba(feature)
        elif hasattr(clf, 'decision_function'):
            predict_proba = clf.decision_function(feature)
        else:
            print("Non-linear model\n")
            predict_proba = prediction

        return predict_proba, prediction

    def evaluate(self, real_label, decision, prediction):
        lcode=preprocessing.LabelEncoder()
        real_label=lcode.fit_transform(real_label)
        prediction = lcode.transform(prediction)
        
        acc = metrics.accuracy_score(real_label, prediction)
        if len(np.unique(real_label)) == 2:
            decision_ = decision[:,-1] if len(np.shape(decision)) > 1 else decision
            fpr, tpr, thresholds = metrics.roc_curve(real_label, decision_, pos_label=np.max(real_label))
            auc = metrics.auc(fpr, tpr)
            f1 = metrics.f1_score(real_label, prediction)
        else:
            auc = np.nan
            f1 = np.nan
        return acc, auc, f1, metrics.confusion_matrix(real_label, prediction)


if __name__ ==  "__main__": 
    model = Model()
    # Denan
    data_train_, label_train_ = model.denan(data_train["AD"], label_train)
    data_val_, label_val_ = model.denan(data_validation["AD"], label_validation)
    
    # Preprocessing
    scaler, data_train_ = model.preprocess_(data_train_)
    data_val_ = scaler.transform(data_val_)

    # Feature selection
    data_train_, label_train_ = model.feature_selection(data_train_, label_train_)
    data_val_, label_val_ = model.feature_selection(data_val_, label_val_)

    # Fit
    clf = model.train_linearSVC(data_train_, label_train_)
    
    # Predict
    predict_proba, prediction = model.predict(clf,data_val_)
    
    # Evaluation
    acc, auc, f1, confmat = model.evaluate(label_val_, predict_proba, prediction)
    
    
    
