# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:52:35 2018

@author: lenovo
"""

    
import os
import pandas as pd

savePath=r'I:\BOLDVar\folder'
hcFolder=basic['folder'][basic['诊断']==1].to_excel(os.path.join(savePath,'hc.xlsx'),index=False,header=False)
mddFolder=basic['folder'][basic['诊断']==2].to_excel(os.path.join(savePath,'mdd.xlsx'),index=False,header=False)
szFolder=basic['folder'][basic['诊断']==3].to_excel(os.path.join(savePath,'sz.xlsx'),index=False,header=False)
bdFolder=basic['folder'][basic['诊断']==4].to_excel(os.path.join(savePath,'bd.xlsx'),index=False,header=False)

import copySelectedFile_OsWalk4 as copy
# basic['folder'].to_csv(r'I:\dynamicALFF\folder.txt',header=False,index=False)                     
sel=copy.copy_fmri(referencePath=os.path.join(savePath,'bd.xlsx'),
              regularExpressionOfsubjName_forReference='([1-9]\d*)',
              folderNameContainingFile_forSelect='',
              num_countBackwards=1,
              regularExpressionOfSubjName_forNeuroimageDataFiles='([1-9]\d*)',\
              keywordThatFileContain='nii',
              neuroimageDataPath=r'I:\BOLDVar\all_BOLDVar',
              savePath=r'I:\BOLDVar\BD',
              n_processess=10,
              ifSaveLog=0,
              ifCopy=0,
              ifMove=1,
              saveInToOneOrMoreFolder='saveToOneFolder',
              saveNameSuffix='',
              ifRun=1)

allFilePath,allSubjName,\
logic_loc,allSelectedFilePath,allSelectedSubjName=\
sel.main_run()
print('Done!')