# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:26:23 2018
将各个窗的状态信息，根据不同诊断分组
@author: lenovo
"""
from moveScreenedFile import moveMain
import pandas as pd
import os
##  ==========================input====================================
# source data
rootpath=r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Data\zDynamic\state'
stateName=['allState17_4','allState17_5','allState17_8',
           'allState20_2','allState20_4','allState20_5','allState20_8']
metricsName=['fractionOfTimeSpentInEachDtate','fullTransitionMatrix','meanDwellTimeInEachState','numberOfTransitions']
##
rootPath=[]
for statename in stateName:
    [rootPath.append(os.path.join(rootpath,statename,metricsname)) for metricsname in metricsName]
#os.mkdir(rootPath[0])
# out path
outPath=[]
for statename in stateName:
    [outPath.append(os.path.join(rootpath,statename,metricsname+'1')) for metricsname in metricsName]
#os.mkdir(outPath[0])
##  ==========================import===================================
from basicInfoStat import folder1,folder2,folder3,folder4
##  ============================run====================================
reguForExtractFileName='[1-9]\d*'

for (rootpath,outpath) in zip(rootPath,outPath):
    screenedFilePath1,logic_loc_refrence1=\
    moveMain(rootpath,folder1,reguForExtractFileName,\
             os.path.join(outpath,'HC'),ifMove=1,ifSaveMoveLog=0)

    screenedFilePath1,logic_loc_refrence1=\
    moveMain(rootpath,folder2,reguForExtractFileName,\
             os.path.join(outpath,'MDD'),ifMove=1,ifSaveMoveLog=0)
    
    screenedFilePath1,logic_loc_refrence1=\
    moveMain(rootpath,folder3,reguForExtractFileName,\
             os.path.join(outpath,'SZ'),ifMove=1,ifSaveMoveLog=0)
    
    screenedFilePath1,logic_loc_refrence1=\
    moveMain(rootpath,folder4,reguForExtractFileName,\
             os.path.join(outpath,'BD'),ifMove=1,ifSaveMoveLog=0)


rootpath=r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Data\zDynamic\state\allState17_2\numberOfTransitions'
outpath=r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Data\zDynamic\state\allState17_2\numberOfTransitions1'

screenedFilePath1,logic_loc_refrence1=\
moveMain(rootpath,folder1,reguForExtractFileName,\
         os.path.join(outpath,'HC'),ifMove=1,ifSaveMoveLog=0)

screenedFilePath1,logic_loc_refrence1=\
moveMain(rootpath,folder2,reguForExtractFileName,\
         os.path.join(outpath,'MDD'),ifMove=1,ifSaveMoveLog=0)

screenedFilePath1,logic_loc_refrence1=\
moveMain(rootpath,folder3,reguForExtractFileName,\
         os.path.join(outpath,'SZ'),ifMove=1,ifSaveMoveLog=0)

screenedFilePath1,logic_loc_refrence1=\
moveMain(rootpath,folder4,reguForExtractFileName,\
         os.path.join(outpath,'BD'),ifMove=1,ifSaveMoveLog=0)