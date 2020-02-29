# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 20:16:14 2018

self-defined  training, validation and test data
rfe-svm-CV
input:
             k=3:k-fold
             step=0.1:rfe step
             num_jobs=1: parallel
             scale_method='StandardScaler':standardization method
             pca_n_component=0.9
             permutation=0
@author: LI Chao
"""
import sys  
import os
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root)

from Utils.lc_featureSelection_rfe import rfeCV
import Utils.lc_dimreduction as pca
import Utils.lc_scaler as scl

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class SVCRFECV():
    def __init__(sel,
                 x_train=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_train380.npy',
                 y_train=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_train380.npy',

                 x_val=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_val76.npy',
                 y_val=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_val76.npy',
                    
                 x_test=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\x_test206.npy',
                 y_test=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\oldCNNdata\data\y_test206.npy',
                 
                 k=5,
                 seed=1,
                 step=0.1,
                 num_jobs=1,
                 scale_method='StandardScaler',
                 pca_n_component=0,
                 permutation=0,
                 show_results=1,
                 show_roc=1):
        
        sel.x_train=x_train
        sel.y_train=y_train
        
        sel.x_val=x_val
        sel.y_val=y_val
        
        sel.x_test=x_test
        sel.y_test = y_test
        
        sel.k=k
        sel.seed=seed # 随机种子
        sel.step=step
        sel.num_jobs=num_jobs
        sel.scale_method=scale_method
        sel.pca_n_component=pca_n_component
        sel.permutation=permutation
        sel.show_results=show_results
        sel.show_roc=show_roc


    def main_svc_rfe_cv(sel):
        
        # 自定义训练集和测试集
        print('training model and testing...\n')
        # load data
        x_train,y_train=sel.load_data_and_label(sel.x_train,sel.y_train,1)
        x_val,y_val=sel.load_data_and_label(sel.x_val,sel.y_val,1)
        x_test,y_test = sel.load_data_and_label(sel.x_test, sel.y_test , 0)
        
        # scale
        x_train=sel.normalization(x_train)
        x_val=sel.normalization(x_val)
        x_test=sel.normalization(x_test)
        
        # pca
        if 0<sel.pca_n_component<1:
            x_train,x_test,trained_pca=sel.dimReduction(x_train,x_test,sel.pca_n_component)
        else:
            pass
        
        # train
        model,weight=sel.training(x_train,y_train,\
             step=sel.step, cv=sel.k,n_jobs=sel.num_jobs,\
             permutation=sel.permutation)
        
        # fetch orignal weight
        sel.weight_all=pd.DataFrame([])
        if 0<sel.pca_n_component<1:
            weight=trained_pca.inverse_transform(weight)
        sel.weight_all=pd.concat([sel.weight_all,pd.DataFrame(weight)],axis=1)
        
        # test
        sel.predict=pd.DataFrame([])
        sel.decision=pd.DataFrame([])
        
        prd,de=sel.testing(model,x_test)
        prd=pd.DataFrame(prd)
        de=pd.DataFrame(de)
        sel.predict=pd.concat([sel.predict,prd])
        sel.decision=pd.concat([sel.decision,de])
         
        # 打印并显示模型性能
        if sel.show_results:
            sel.eval_prformance(sel.decision, y_test, sel.predict)
        return  sel
    
    def load_data_and_label(sel,x,y,label_col):
        x=np.load(x)
        y=np.load(y)[:,label_col]
        return x,y

    def normalization(sel,data):
        # because of our normalization level is on subject, 
        # we should transpose the data matrix on python(but not on matlab)
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data=scaler.transform(data.T) .T
        return z_data
    
    def scaler(sel,train_X,test_X,scale_method):
        train_X,model=scl.scaler(train_X,scale_method)
        test_X=model.transform(test_X)
        return train_X,test_X
    
    def dimReduction(sel,train_X,test_X,pca_n_component):
        train_X,trained_pca=pca.pca(train_X,pca_n_component)
        test_X=trained_pca.transform(test_X)
        return train_X,test_X,trained_pca
    
    def training(sel,x,y,\
                 step, cv,n_jobs,permutation):
    #    refCV
        model,weight=rfeCV(x,y,step, cv,n_jobs,permutation)
        return model,weight
    
    def testing(sel,model,test_X):
        predict=model.predict(test_X)
        decision=model.decision_function(test_X)
        return predict,decision
    
    def eval_prformance(sel, decision_value, true_label, predict_label):
        # 此函数返回sel
        # transform the label to 0 and 1
        true_label = sel.label_to_binary(true_label)
        predict_label = sel.label_to_binary(predict_label)
        
        # loss OR error
        sel.er=true_label.reshape(len(true_label),1)-decision_value.values.reshape(len(decision_value),1)
        sel.er=np.mean(np.abs(sel.er))
        print('loss=',sel.er)
        
        # accurcay, sel.specificity(recall of negative) and sel.sensitivity(recall of positive)        
        sel.accuracy= accuracy_score (true_label,predict_label.values)
        
        report=classification_report(true_label,predict_label.values)
        report=report.split('\n')
        sel.specificity=report[2].strip().split(' ')
        sel.sensitivity=report[3].strip().split(' ')
        sel.specificity=float([spe for spe in sel.specificity if spe!=''][2])
        sel.sensitivity=float([sen for sen in sel.sensitivity if sen!=''][2])
        
        sel.balanced_accuracy=(sel.specificity+sel.sensitivity)/2
        
        # sel.confusion_matrix matrix
        sel.confusion_matrix=confusion_matrix(true_label,predict_label.values)

        # roc and sel.auc
        fpr, tpr, thresh = roc_curve(true_label,decision_value.values)
        sel.auc=roc_auc_score(true_label,decision_value.values)
        
        # print performances
#        print('混淆矩阵为:\n{}'.format(sel.confusion_matrix))
        
        print('\naccuracy={:.2f}\n'.format(sel.accuracy))
        print('balanced_accuracy={:.2f}\n'.format(sel.balanced_accuracy))
        print('sensitivity={:.2f}\n'.format(sel.sensitivity))
        print('specificity={:.2f}\n'.format(sel.specificity))
        print('auc={:.2f}\n'.format(sel.auc))

        if sel.show_roc:
            fig,ax=plt.subplots()
            ax.plot(figsize=(5, 5))
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True)
            ax.plot(fpr, tpr,'-')
            
            #设置坐标轴在axes正中心
            ax.spines['top'].set_visible(False) #去掉上边框
            ax.spines['right'].set_visible(False) #去掉右边框
#            ax.spines['bottom'].set_position(('axes',0.5 ))
#            ax.spines['left'].set_position(('axes', 0.5))
           
        return sel
    
    def label_to_binary(sel,y):
        unique_y=np.unique(y)
        y[y==np.min(unique_y)]=0
        y[y==np.max(unique_y)]=1
        return y

#        
if __name__=='__main__':
    sel=SVCRFECV(k=5)
    results=sel.main_svc_rfe_cv()
    results=results.__dict__
