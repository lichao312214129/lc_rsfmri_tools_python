# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:38:20 2018
dimension reduction using sklearn
@author: lenovo
"""
from sklearn.feature_selection import VarianceThreshold
import numpy as np
#
X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], \
              [0, 1, 1], [0, 1, 0], [0, 1, 1]])
#
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
