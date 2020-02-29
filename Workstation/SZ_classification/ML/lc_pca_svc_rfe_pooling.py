# -*- coding: utf-8 -*-
"""
Created on 2019/11/20
This script is used to training a linear svc model using a given training dataset, and validation this model using validation dataset.
Finally, we test the model using test dataset.
NOTE: We use PCA+SVC-RFE to select features.
@author: LI Chao
"""
import sys  
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
# sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python\Utils')
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import Utils.lc_dimreduction as dimreduction


class SVCRFECV():
    def __init__(sel,
                 dataset_our_center_414=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_414_from_550.npy',
                 dataset_our_center_136=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_136_from_550.npy',
                 dataset_206=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_206.npy',
                 dataset_COBRE=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_COBRE.npy',
                 dataset_UCAL=r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\ML_data_npy\dataset_UCLA.npy',
                 is_dim_reduction=1,
                 components = 0.95,
                 cv=3,
                 step=0.2,
                 show_results=1,
                 show_roc=1):
        sel.dataset_our_center_414=dataset_our_center_414
        sel.dataset_our_center_136=dataset_our_center_136
        sel.dataset_206=dataset_206
        sel.dataset_COBRE=dataset_COBRE
        sel.dataset_UCAL=dataset_UCAL

        sel.is_dim_reduction=is_dim_reduction
        sel.components = components
        sel.cv=cv
        sel.step = step
        sel.show_results=show_results
        sel.show_roc=show_roc


    def main_svc_rfe_cv(sel):
        # the training data, validation data and  test data are randomly splited
        # load data
        dataset_our_center_414 = np.load(sel.dataset_our_center_414)
        dataset_our_center_136 = np.load(sel.dataset_our_center_136)
        dataset_206 = np.load(sel.dataset_206)
        dataset_COBRE = np.load(sel.dataset_COBRE)
        dataset_UCAL = np.load(sel.dataset_UCAL)

        # Extracting features and label
        features_our_center_414 = dataset_our_center_414[:,2:]
        features_our_center_136 = dataset_our_center_136[:,2:]
        features_206 = dataset_206[:,2:]
        features_COBRE = dataset_COBRE[:,2:]
        features_UCAL = dataset_UCAL[:,2:]

        label_our_center_414 = dataset_our_center_414[:,1]
        label_our_center_136 = dataset_our_center_136[:,1]
        label_206 = dataset_206[:,1]
        label_COBRE = dataset_COBRE[:,1]
        label_UCAL = dataset_UCAL[:,1]

        uid_our_center_414 = dataset_our_center_414[:,0]
        uid_our_center_136 = dataset_our_center_136[:,0]
        
        # Generate training data and test data
        feature_train = np.concatenate([features_our_center_414, features_206, features_COBRE, features_UCAL], axis=0)
        label_train = np.concatenate([label_our_center_414, label_206, label_COBRE, label_UCAL],axis=0)

        feature_test = features_our_center_136
        label_test = label_our_center_136
		
        # template test
        # feature_all = np.concatenate([features_our_center_414, features_our_center_136, features_COBRE, features_UCAL, features_206], axis=0)
        # label_all = np.concatenate([label_our_center_414, label_our_center_136, label_COBRE, label_UCAL, label_206],axis=0)

        # np.random.seed(1)
        # randomint = np.random.permutation(range(0,len(label_all)))
        # iloc_train = randomint[:900]
        # iloc_test = randomint[900:]
        # feature_train = feature_all[iloc_train]
        # label_train = label_all[iloc_train]
        # feature_test = feature_all[iloc_test]
        # label_test = label_all[iloc_test]

        # resampling training data
        feature_train, label_train = sel.re_sampling(feature_train, label_train)

        # normalization
        feature_train = sel.normalization(feature_train)
        feature_test = sel.normalization(feature_test)

        # dimension reduction
        if sel.is_dim_reduction:
            feature_train, feature_test, model_dim_reduction = sel.dimReduction(feature_train, feature_test, sel.components)
            print(f'After dimension reduction, the feature number is {feature_train.shape[1]}')
        else:
            print('No dimension reduction perfromed\n')
        # Training
        print('training model...\n')
        model,sel.coef = sel.svc_rfe_CV(feature_train,label_train, 0.2, sel.cv, 4)

        if sel.is_dim_reduction:
            sel.coef = model_dim_reduction.inverse_transform(sel.coef)
        else:
            pass
        print('testing model...\n')
        sel.prediction,sel.decision=sel.testing(model,feature_test)
        sel.prediction = np.array(sel.prediction)
        sel.decision = np.array(sel.decision)
        # Evaluating classification performances
        mse, accuracy, sensitivity, specificity, auc = sel.eval_prformance(sel.decision, label_test, sel.prediction)  
        print(f'performances = {accuracy,sensitivity, specificity,auc}')
        return  sel
	
    def re_sampling(sel,feature, label):
        """
        Used to over-sampling unbalanced data
        """
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter
        print(f'Each orignal label number is {sorted(Counter(label).items())}')
        ros = RandomOverSampler(random_state=0)
        feature_resampled, label_resampled = ros.fit_resample(feature, label)
        print(f'Each resampleed label number is {sorted(Counter(label_resampled).items())}')
        return feature_resampled, label_resampled

    def normalization(sel, data):
        '''
        Because of our normalization level is on subject, 
        we should transpose the data matrix on python(but not on matlab)
        '''
        scaler = preprocessing.StandardScaler().fit(data.T)
        z_data=scaler.transform(data.T) .T
        return z_data

    def dimReduction(self,train_X,test_X, pca_n_component):
        train_X, trained_pca = dimreduction.pca(train_X,pca_n_component)
        test_X = trained_pca.transform(test_X)
        return train_X, test_X, trained_pca

    def svc_rfe_CV(sel, train_X, train_y, step, cv, n_jobs):
        """equal to nested rfe"""
        print(f'Performing SVC based RFECV...\n')
        from sklearn.feature_selection import RFECV
        n_samples, n_features = train_X.shape
        estimator = svm.SVC(kernel="linear", C=1, class_weight='balanced', max_iter=5000, random_state=0)
        selector = RFECV(estimator, step=sel.step, cv=cv, n_jobs=n_jobs)
        selector = selector.fit(train_X, train_y)
        mask = selector.support_
        optmized_model = selector.estimator_
        w = optmized_model.coef_  # when it is multi-class classification, the w is two dimensional
        weight = np.zeros([w.shape[0], n_features])
        weight[:, mask] = w
    #    selector.score(x, y)
    #    y_pred=selector.predict(x)
    #    r=np.corrcoef(y,y_pred)[0,1]
        return selector, weight

    def testing(sel,model,test_X):
        predict = model.predict(test_X)
        decision = model.decision_function(test_X)
        return predict,decision
    
    def eval_prformance(sel, decision_value, true_label, predict_label, verbose=0):
        # 此函数返回sel
        # transform the label to 0 and 1
        
        # loss OR error
        mse = mean_squared_error(decision_value, true_label)

        # accurcay, specificity(recall of negative) and sensitivity(recall of positive)        
        accuracy= accuracy_score (true_label,predict_label)
        
        report=classification_report(true_label,predict_label)
        report=report.split('\n')
        specificity=report[2].strip().split(' ')
        sensitivity=report[3].strip().split(' ')
        specificity=float([spe for spe in specificity if spe!=''][2])
        sensitivity=float([sen for sen in sensitivity if sen!=''][2])
        
        balanced_accuracy=(specificity+sensitivity)/2
        
        # sel.confusion_matrix matrix
        confus_mat = confusion_matrix(true_label,predict_label)

        # roc and auc
        fpr, tpr, thresh = roc_curve(true_label,decision_value)
        auc=roc_auc_score(true_label,decision_value)
        
        # print performances
#        print('混淆矩阵为:\n{}'.format(confusion_matrix))
        if verbose:
            print('\naccuracy={:.2f}\n'.format(accuracy))
            print('balanced_accuracy={:.2f}\n'.format(balanced_accuracy))
            print('sensitivity={:.2f}\n'.format(sensitivity))
            print('specificity={:.2f}\n'.format(specificity))
            print('auc={:.2f}\n'.format(auc))

            if sel.show_roc:
                fig,ax=plt.subplots()
                ax.plot(figsize=(5, 5))
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.grid(True)
                ax.plot(fpr, tpr,'-')
                # 
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False)
        return mse, accuracy, sensitivity, specificity, auc

#        
if __name__=='__main__':
    sel=SVCRFECV()
    results=sel.main_svc_rfe_cv()
    results=results.__dict__
