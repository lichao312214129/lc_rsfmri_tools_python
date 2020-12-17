# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:43:15 2020

@author: lenovo
"""

a  # 因为a没有赋值，所以会报错

# 用try except来达到容错的目的，同时也获取错误信息
try:
    a
except Exception as e:
    print(f'程序出错:{e}') # 程序出错 list index out of range
    a = 1

print(f'a被赋值，其值为{a}\n')
del a


# 多个exception
try:
    a
    
except ValueError:
    print('ValueError') # 程序出错 list index out of range
    
except NameError:
    print('NameError')
    
    