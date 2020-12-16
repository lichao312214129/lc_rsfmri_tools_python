# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:56:14 2020
元组对于列表来说，元组不允许增、删、改，但可以查和删除整个元组
元组起到保护数据作用
@author: Li Chao
"""


# 生成元组
a = ('i', 'love', 'python', 1, 3, 1, 4)

# 查看类型
type(a)

# 假如元组只有一个元素，后面加逗号，否则python视其为该元素类型
b = (1,)
type(b)

b = (1)
type(b)


# 元组不允许增、删、改
a.append("do you like python")
del a[-1]  
a.pop(-1) 
a.remove("do you like python")  # 按内容删除
a[0] = "you"

# 查
a[-1]
a[6]

# 可以删除整个元组
del a
