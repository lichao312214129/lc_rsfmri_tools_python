# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:29:27 2018

@author: lenovo
"""

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

t,p=stats.ttest_ind(a,b)
aa=np.mean(a1.values,axis=0)
bb=np.mean(a2.values,axis=0)
a=np.corrcoef(aa,bb)
    
for i in range(5):

    fig,ax=plt.subplots()
    n, bins, patches =ax.hist(x=a[:,32],bins='auto',rwidth=0.9,alpha=0.5,color='b')
    n, bins, patches =ax.hist(x=b[:,32],bins='auto',rwidth=0.9,alpha=0.5,color='r')
