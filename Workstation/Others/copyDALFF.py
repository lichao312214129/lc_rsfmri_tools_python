# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:10:51 2018
条件：subjxxx下有多个文件，其中只有一个是我们需要的
目的：将所有被试文件夹下，这一个我们需要的文件拷贝到另一个文件夹中
同时每一个文件都被名字为subjxxx的文件夹存放
@author: lenovo
"""
import sys
sys.path.append(r'D:\myCodes\MVPA_LIChao\MVPA_Python\workstation')
import copySelectedFile_OsWalk3 as copy
# ====================================================================
# input
referenceFile=(r'D:\WorkStation_2018\WorkStation_2018_08_Doctor_DynamicFC_Psychosis\Data\zDynamic\state\subjName.xlsx')
# basic['folder'].to_csv(r'I:\dynamicALFF\folder.txt',header=False,index=False)                     
sel=copy.copy_fmri(referencePath=referenceFile,
                  regularExpressionOfsubjName_forReference='([1-9]\d*)',
                  folderNameContainingFile_forSelect='',
                  num_countBackwards=2,
                  regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',\
                  keywordThatFileContain='var_mdALFF',
                  neuroimageDataPath=r'I:\dynamicALFF\Results\DALFF\80_0.9\variance_dALFFmap',
                  savePath=r'I:\dynamicALFF\Results\DALFF\80_0.9',
                  saveFolderName='variance_dmALFF',
                  n_processess=10,
                  ifSaveLog=0,
                  ifCopy=1,
                  ifMove=1,
                  saveInToOneOrMoreFolder='saveToOneFolder',
                  saveNameSuffix='.nii',
                  ifRun=1)
 
allFilePath,allSubjName,logic_loc,allSelectedFilePath,allSelectedSubjName=\
                                                             sel.main_run() 