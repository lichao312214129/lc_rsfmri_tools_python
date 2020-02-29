# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:01:58 2018
0 把allContent的所有空列表删除
1 获得每一个text文档中所有投资人的名单(某个项目的所有内容)，并组成一个列表
2 获得unique名单
3 目的是为了方便后续的整理 汇总做准备
input:
    allContent:一个列表，里面保存每个text文档的数据
    itemName：项目名称，次函数的名称应该为'投资人'
output:
    unique_allName_1d：不重复的所有text包含的投资人姓名
    allName_noSpace：所有text中的投资人姓名（M*N；M为text数目，N为每个text中行数）
@author: lenovo
"""

## import
import numpy as np
import operator 
from functools import reduce

##
def obtainOneItemContent(allContent,itemName):
    for i in range(allContent.count([])):
        allContent.remove([])
      #-------减号之间的为可选项-------
    allName=[]
    for content in allContent:
        allName.append([a['投资人'].iloc[0] for a in content])
        # 把allName的空格去掉
    allName_noSpace=[]
    for Name in allName:
        allName_noSpace.append([name.replace(' ','') for name in Name])
        # 把all name拉成一列，并去掉空格，最后求unique name
    allName_1d=reduce(operator.add, allName)
    allName_1d=[name.replace(' ','') for name in allName_1d]
    unique_allName_1d=np.unique(allName_1d) # 所有不重复的需要提取的人名
    return unique_allName_1d,allName_noSpace