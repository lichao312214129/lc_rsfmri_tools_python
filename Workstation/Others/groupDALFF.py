# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:10:51 2018
将dmALFF分组
@author: lenovo
"""
import sys
import pandas as pd
#sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\workstation')
import copySelectedFile_OsWalk4 as copy

# ====================================================================
# input
referenceFile_HC=(r'H:\dynamicALFF\Scales\folder_BD.xlsx')
#referenceFile_MDD=(r'H:\dynamicALFF\folder2.xlsx')
#referenceFile_SZ=(r'H:\dynamicALFF\folder3.xlsx')
#referenceFile_BD=(r'H:\dynamicALFF\folder4.xlsx')
#referenceFile=[referenceFile_HC,referenceFile_MDD,
#               referenceFile_SZ,referenceFile_BD]
#subjName_forSelect=pd.read_excel(referenceFile_HC,dtype='str',header=None)
# ============================================================================        
sel=copy.copy_fmri(referencePath=referenceFile_HC,
                  regularExpressionOfsubjName_forReference='([1-9]\d*)',
                  folderNameContainingFile_forSelect='',
                  num_countBackwards=1,
                  regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',\
                  keywordThatFileContain='nii',
                  neuroimageDataPath=r'H:\dynamicALFF\Results\DALFF\50_0.9\BD_smooth',
                  savePath=r'H:\dynamicALFF\Results\DALFF\50_0.9\BD_smooth_screened',
                  n_processess=10,
                  ifSaveLog=0,
                  ifCopy=0,
                  ifMove=1,
                  saveInToOneOrMoreFolder='saveToOneFolder',
                  saveNameSuffix='',
                  ifRun=1)
 
allFilePath,allSubjName,logic_loc,allSelectedFilePath,allSelectedSubjName=\
                                                                 sel.main_run() 