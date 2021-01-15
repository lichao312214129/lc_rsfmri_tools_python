# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:37:40 2020

@author: lenovo
"""

import numpy as np

# 生成列表
a = ['i', 'love', 'python', 1, 3, 1, 4]

a = np.array(a)

idx = np.array((2,3))
a[[2,3]]

a = list(a)

# 查看类型
type(a)

# 增
a.append("do you like python")
a.extend(["do you like python"])

# 删
del a[-1]  # 按照位置删除
print(a)
a.append("do you like python")

a.pop(-1) # 按照位置删除
print(a)
a.append("do you like python")

a.remove("do you like python")  # 按内容删除
print(a)
a.append("do you like python")

# 查
a[-1]
a[7]

# 改
a[0] = "you"
print(a)
a[2:6] = ["matlab", 1,3,1]

# 作业1：一次性查看列表第3和第4个元素
# 作业2：一次性查看列表第3和第5个元素