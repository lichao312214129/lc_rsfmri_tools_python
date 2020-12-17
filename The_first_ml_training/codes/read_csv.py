# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:45:57 2020

@author: lenovo
"""

import pandas as pd


data_file = '../demo_data/winequality-red.csv'

data = pd.read_csv(data_file, sep=';')

# 描述数据
describe = data.describe()

# 查看数据的列名
colname = data.columns

# 获取数据数据的值
values = data.values



