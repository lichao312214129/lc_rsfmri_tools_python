# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:02:52 2018

@author: lenovo
"""
import pandas as pd

scaleDataPath=r"D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Scales\8.30大表.xlsx"
trainDataPath=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\trainingData.xlsx'
testScalePath=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\REST-meta-MDD-PhenotypicData_WithHAMDSubItem_S20.xlsx'
predictLabelPath=r'D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\predictLabel_testData.xlsx'
#
scale=pd.read_excel(scaleDataPath)
trainData=pd.read_excel(trainDataPath)
testScale=pd.read_excel(testScalePath)
predictLabel=pd.read_excel(predictLabelPath,header=None,index=False)
predictLabel.columns=['label']

#
testScale_predictLabel=pd.concat([testScale,predictLabel],axis=1)
testScale_predictLabel.to_excel('testsScale_plus_predictLabel.xlsx')

# 
joinDf=scale.set_index('folder').join(trainData.set_index('folder'),how='right')
diagnosis=joinDf['诊断']

#