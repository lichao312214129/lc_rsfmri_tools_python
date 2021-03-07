# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:04:31 2021
此代码用来最终生成被赋值的.annot文件
@author: Li Chao
Email: lichao19870617@163.com
"""

import nibabel as nib
import pandas as pd
import numpy as np

# ============All input============
annotFile = "./lh.HCP-MMP1.annot"  # ***模板
valueFile = "./脑区值.xlsx"  # ***需要呈现的值
sheetName = "Sheet2"  # ***需要赋值的值在annotFile的哪个表格
mapFile = "./map.xlsx"  # ***通过get_colormap.m生成的colormap文件目录
# =================================

# Load value and colormap
value = pd.read_excel(valueFile, sheet_name=sheetName, header=None)
mapData = pd.read_excel(mapFile, header=None)

value.fillna(value=0, inplace=True)
value = value[:180]  # ***前180个值是左脑的，如果是右脑则为value[180:]， 请自行修改
uv = value[1].unique()
sortUv = np.sort(uv)

# Load annot file
annot = nib.freesurfer.read_annot(annotFile)

# Generate annot background color
annotColor = np.repeat([255,255,255,255], 180).reshape(180,4)
annotColor = np.int32(annotColor)

# Give color to annotColor according to its value in corresponding index
for uv_ in uv:
    cvalue = mapData.iloc[np.where(np.in1d(sortUv, uv_))[0][0],:].values
    cvalue = np.float32(cvalue)*255
    if uv_ != 0:
        cvalue = np.hstack([cvalue, 0])
    else:
        cvalue = [255,255,255,255]
    loc = np.array(value[value[1] == uv_].index)[0]
    annotColor[loc, [0,1,2,3]] = cvalue
annotColor = np.int32(annotColor)

annot_ = list()
annot_.append(annot[0])
annot_.append(annotColor)
annot_.append(annot[2])
annot1_ = annot[1]
annot1_[1:,[0,1,2,3]] = annotColor
annot_[1] = annot1_

# Save
nib.freesurfer.write_annot("./values.annot", *annot_)
annot2 = nib.freesurfer.read_annot("./values.annot")
