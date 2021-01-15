# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:54:14 2020

@author: lenovo
"""

# 生成列表
hanmeimei = {"gender": "女"}
a = dict()

# 查看类型
type(hanmeimei)

# 增
hanmeimei.update({"height":165})  # 添加一个键值对
print(hanmeimei)

hanmeimei["学历"] = "Phd"  # 通过给一个新的键添加值
print(hanmeimei)

# 删
del hanmeimei["height"]  # del
print(hanmeimei)

hanmeimei.update({"height":165})  

hanmeimei.pop("height") # pop
print(hanmeimei)

hanmeimei.update({"height":165}) 


# 查
hanmeimei["学历"]

# 改
hanmeimei["height"] = 168
print(hanmeimei)

import pandas as pd
import numpy as np

d = np.random.randn(10,3)
d = pd.DataFrame(d)
d.rename(columns={0: "Trial"})
onset_index, offset_index = [0,3], [4,8]
data = d

data['Trial'] = np.nan
for i,j in enumerate(zip(onset_index, offset_index)):
    data['Trial'].iloc[j[0]:j[1]] = i

