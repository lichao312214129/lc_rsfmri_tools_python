# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:54:14 2020

@author: lenovo
"""

# 生成列表
hanmeimei = {"gender": "女"}

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
hanmeimei[-1]
hanmeimei[7]

# 改
hanmeimei["height"] = 168
print(hanmeimei)
