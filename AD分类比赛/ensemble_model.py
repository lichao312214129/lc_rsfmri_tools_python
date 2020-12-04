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
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


class Model(object):

    def __init__(self):
        self.file = r'F:\AD分类比赛\MCAD_AFQ_competition.mat'
        self.seed = 666
        # self.pca_n_component = np.linspace(0.7, 0.99, 5)
        self.pca_n_component = 0.95
        # self.regularization_strength = np.logspace(-4, 2, 10)
        self.regularization_strength = 0.0001
    
    def denan(self, data, label, fill=True):
        """
        De-nan 
        Parameters:
        -----------
        data: ~

        label: ~

        fill: Boolean
            If True then fill nan with values (hear is mean of each feature)
            if False then drop whole feature of one case if any nan in this case.
        """


        if not isinstance(data, pd.core.frame.DataFrame):
                data = pd.DataFrame(data)
        
        value = data.mean()
        if fill:
            data_ = data.fillna(value=value)
            label_ = label
        else:       
            idx_nan = np.sum(np.isnan(data),axis=1) > 0  # Drop the case if have any nan
            data_ = data.dropna().values
            label_ = label[idx_nan == False]
            
        return data_, label_, value
    
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
    
    def feature_selection(self, data, label):
        selector = SelectPercentile(f_classif, percentile=80)
        selector.fit(data, label)
        data_  = selector.transform(data)
        return selector, data_

    def make_xgboost(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),  # C=0.014
         ('classify', XGBClassifier(learning_rate=0.1, alpha=22, objective='binary:logistic', random_state=self.seed)),
         # ('classify', XGBClassifier(learning_rate=0.1, alpha=14.3, objective='binary:logistic', random_state=self.seed)),
        # ('classify', XGBClassifier(learning_rate=0.1, alpha=22, objective='binary:logistic', random_state=self.seed)),
        ])
        return pipe

    def make_ridge_regression(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(C=0.0138, penalty="l1", dual=False))),
        # ('feature_selection', SelectPercentile(f_classif, percentile=96)),
        # ('reduce_dim', PCA(n_components=0.95, random_state=self.seed)),
        ('classify', RidgeClassifier(alpha=1, random_state=self.seed)),
        ])

        # param_grid = {
        #     'feature_selection__percentile': [85],
        #     'reduce_dim__n_components': self.pca_n_component,
        #     'classify__alpha': self.regularization_strength
        # }
        
        # grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
        #                     scoring='accuracy', refit=True)
        return pipe

    def make_logistic_regression(self):
        pipe = Pipeline([
        # ('feature_selection', SelectPercentile(f_classif, percentile=80)),
        ('reduce_dim', PCA(n_components=0.95, random_state=self.seed)),
        ('classify', LogisticRegression(C= 0.005, random_state=self.seed)),
        ])

        # param_grid = {
        #     'feature_selection__percentile': [80],
        #     'reduce_dim__n_components': self.pca_n_component,
        #     'classify__C': self.regularization_strength
        # }
        
        # grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
        #                     scoring='accuracy', refit=True)
        return pipe

    def make_linearSVC(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=False))),
        # ('reduce_dim', PCA(n_components=0.99, random_state=self.seed)),
        ('classify', LinearSVC(C=0.00003, random_state=self.seed)),
        ])

        param_grid = {
            # 'feature_selection__percentile': [60],
            # 'reduce_dim__n_components': self.pca_n_component,
            'classify__C': [0.00001, 0.00003, 0.00006, 0.00009, 0.0001]
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)
        
        return pipe
    
    def make_SVC(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        # ('reduce_dim', PCA(n_components=0.95, random_state=self.seed)),
        ('classify', SVC(C=0.5, random_state=self.seed, kernel='sigmoid')),
        ])

        # param_grid = {
        #     # 'feature_selection__percentile': [70],
        #     'reduce_dim__n_components': self.pca_n_component,
        #     'classify__C': [0.5]
        # }
        
        # grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
        #                     scoring='accuracy', refit=True)
        return pipe
    
    def make_SVC_rbf(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LogisticRegression())),
        # ('reduce_dim', PCA(n_components=0.95, random_state=self.seed)),
        ('classify', SVC(C=1, random_state=self.seed, kernel='rbf')),
        ])

        # param_grid = {
        #     # 'feature_selection__percentile': [70],
        #     'reduce_dim__n_components': self.pca_n_component,
        #     'classify__C': [0.5]
        # }
        
        # grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
        #                     scoring='accuracy', refit=True)
        return pipe
    
    def make_SVC_poly(self):
        pipe = Pipeline([
        # ('feature_selection', SelectFromModel(LinearSVC())),
        ('reduce_dim', PCA(n_components=0.9, random_state=self.seed)),
        ('classify', SVC(C=0.7, random_state=self.seed, kernel='poly')),
        ])

        # param_grid = {
        #     # 'feature_selection__percentile': [70],
        #     'reduce_dim__n_components': self.pca_n_component,
        #     'classify__C': [0.5]
        # }
        
        # grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
        #                     scoring='accuracy', refit=True)
        return pipe

    def make_mlp(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        # ('reduce_dim', PCA(n_components=0.95, random_state=self.seed)),
        ('classify', MLPClassifier(solver='adam', epsilon=1e-5, max_iter=100, batch_size=40,
                                   alpha=10, hidden_layer_sizes=(100, 50), random_state=self.seed)),
        ])
        
        
        param_grid = {
            'classify__alpha': [10,12,13],
        }
        
        grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid,
                            scoring='accuracy', refit=True)
        
        return pipe
    
    def make_gnb(self):
        pipe = Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(C=0.0138, penalty="l1", dual=False))),
        # ('reduce_dim', PCA(n_components=0.70, random_state=self.seed)),
        ('classify', GaussianNB()),
        ])
        return pipe
        
    def train_ensemble_classifier(self, feature, label, *args):
        """Using sklearn ensemble methods to train multiple classifiers

        Parameters:
        ----------
        feature: ~
        label: ~
        *args: original models

        Returns:
        --------
        clf: trained voting model
        """
        
        if len(args) == 1:
            clf = args[0]
        else:
            name_model = [(str(i),args_) for i,args_ in enumerate(args)]
            clf = StackingClassifier(estimators=name_model, final_estimator=LogisticRegression(), n_jobs=2)
            
        # Fit ensemble model
        clf.fit(feature, label)
        return clf

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
        # TODO: Standardize the predict_proba for merging
        scaler = StandardScaler()
        predict_proba = scaler.fit_transform(predict_proba)
        predict_proba = np.mean(predict_proba, axis=1)
        
        prediction = np.int16(predict_proba > 0)
        
        # prediction = [pred[1] for pred in predict_results]
        # prediction = pd.DataFrame(prediction).T.values
        # prediction = np.array([np.argmax(np.bincount(prediction[i,:])) for i in range(len(prediction))])
    
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
        report = metrics.classification_report(real_label, prediction)
        acc = metrics.accuracy_score(real_label, prediction)
        if len(np.unique(real_label)) == 2:
            decision_ = decision[:,-1] if len(np.shape(decision)) > 1 else decision
            fpr, tpr, thresholds = metrics.roc_curve(real_label, decision_, pos_label=np.max(real_label))
            auc = metrics.auc(fpr, tpr)
            f1 = metrics.f1_score(real_label, prediction)
        else:
            auc = np.nan
            _,_,f1,_ = metrics.precision_recall_fscore_support(real_label, prediction)
        return acc, auc, f1, metrics.confusion_matrix(real_label, prediction), report
    
    
