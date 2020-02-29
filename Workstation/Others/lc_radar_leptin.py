# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:48:46 2018

@author: lenovo
"""
import pandas as pd
import numpy as np

file=r'D:\others\彦鸽姐\leptin-83-figure-data.xlsx'
data=pd.read_excel(file)

#hc=data[data['诊断']==1]['Leptin'].to_excel('D:\others\彦鸽姐\leptin_hc.xlsx',index=False)
#mdd=data[data['诊断']==2]['Leptin'].to_excel('D:\others\彦鸽姐\leptin_mdd.xlsx',index=False)
#scz=data[data['诊断']==3]['Leptin'].to_excel('D:\others\彦鸽姐\leptin_scz.xlsx',index=False)
#bpd=data[data['诊断']==4]['Leptin'].to_excel('D:\others\彦鸽姐\leptin_bpd.xlsx',index=False)

hc=data[data['诊断']==1]['Leptin']
mdd=data[data['诊断']==2]['Leptin']
scz=data[data['诊断']==3]['Leptin']
bpd=data[data['诊断']==4]['Leptin']

mean=[np.mean(hc),np.mean(mdd),np.mean(scz),np.mean(bpd)]
