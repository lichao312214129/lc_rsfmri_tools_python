# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:17:14 2021
This code is used to prepare the colormap which is the presentation of values
@author: Li Chao
Email: lichao19870617@163.com
"""

import nibabel as nib
import pandas as pd
import numpy as np

# ============All input============
valueFile = "./脑区值.xlsx"  # ***需要呈现的值
sheetName = "Sheet2"  # ***需要赋值的值在annotFile的哪个表格
# =================================

# Load value and colormap
value = pd.read_excel(valueFile, sheet_name=sheetName, header=None)
value = value[:180]  # ***前180个值是左脑的，如果是右脑则为value[180:]， 请自行修改
uv = value[1].unique()
print(f"Length of unique value = {len(uv)}")
