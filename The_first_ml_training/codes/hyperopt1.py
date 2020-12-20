# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:17:27 2020

@author: Li Chao
Email: lichao19870617@163.com
"""

#%%
import numpy as np
from sklearn import  datasets
from sklearn.svm import  SVC
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import fmin, tpe, hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt

best = fmin(
    fn=lambda x: x,
    space=hp.uniform('x', 0, 1),
    algo=tpe.suggest,
    max_evals=100)

print (best)


#%%
# 用sklearn生成分类数据
X, y = datasets.make_classification(n_samples=500, n_features=500, n_informative=10, random_state=666)

# 拆分成训练集和测试集
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4, random_state=666)

def hyperopt_train_test(params, x, y):
    clf = SVC(**params)
    return cross_val_score(clf, x, y, cv=3).mean()

space4svm = {
    'C': hp.uniform('C', 0, 5),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid']),
    'gamma': hp.uniform('gamma', 0, 10),
}

def f(params):
    acc = hyperopt_train_test(params, X_train, y_train)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
print (best)
best["kernel"] = "linear"

notbest = best.copy()
notbest["C"] = 1
notbest["gamma"] = 2
notbest["kernel"] = "sigmoid"
hyperopt_train_test(notbest, X_test, y_test)

hyperopt_train_test(best, X_test, y_test)
 
# parameters = ['C', 'kernel', 'gamma']
# cols = len(parameters)
# f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
# cmap = plt.cm.jet
# for i, val in enumerate(parameters):
#     xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
#     ys = [-t['result']['loss'] for t in trials.trials]
#     xs, ys = zip(*sorted(zip(xs, ys)))
#     axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=1)
#     axes[i].set_title(val)
#     axes[i].set_ylim([0.9, 1.0])