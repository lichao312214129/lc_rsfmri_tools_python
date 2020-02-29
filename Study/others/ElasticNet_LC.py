# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:38:08 2018
执行elastic net 回归分析，主要用于MVPA
@author: copy from network, but revised by LiChao
"""
import numpy as np
from sklearn import linear_model

# Generate sample data
n_samples_train, n_samples_test, n_features = 75, 150, 500
np.random.seed(0)
coef = np.random.randn(n_features)
coef[50:] = 0.0  # only the top 10 features are impacting the model
X = np.random.randn(n_samples_train + n_samples_test, n_features)
y = np.dot(X, coef)
 
# Split train and test data
X_train, X_test = X[:n_samples_train], X[n_samples_train:]
y_train, y_test = y[:n_samples_train], y[n_samples_train:]

# Compute train and test errors
def elasticNet_LC(data,label,\
                  alphas=np.logspace(-5, 1, 60),\
                  l1_ratio=0.5):
    enet = linear_model.ElasticNet(l1_ratio=l1_ratio)
    train_errors = list()
    test_errors = list()
    for alpha in alphas:
        enet.set_params(alpha=alpha,random_state=0)
        enet.fit(data, label)
        train_errors.append(enet.score(data, label))
        test_errors.append(enet.score(X_test, y_test))
     
    i_alpha_optim = np.argmax(test_errors)
    alpha_optim = alphas[i_alpha_optim]
    print("Optimal regularization parameter : %s" % alpha_optim)
     
    # Estimate the coef_ on full data with optimal regularization parameter
    enet.set_params(alpha=alpha_optim)
    coef_ = enet.fit(X, y).coef_
     
    ###############################################################################
    # Plot results functions
     
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.semilogx(alphas, train_errors, label='Train')
    plt.semilogx(alphas, test_errors, label='Test')
    plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
               linewidth=3, label='Optimum on test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.2])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Performance')
     
    # Show estimated coef_ vs true coef
    plt.subplot(2, 1, 2)
    plt.plot(coef, label='True coef')
    plt.plot(coef_, label='Estimated coef')
    plt.legend()
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
    plt.show()