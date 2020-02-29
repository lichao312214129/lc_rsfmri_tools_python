# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:16:37 2018
Turning the path into a python readable path
@author: lenovo
"""
# import 
import re
path='D:\myCodes\MVPA_LC\Python\study'
def path_transform(target_str='s',orignal_str):
    news=re.sub(target_str,transformed_path,orignal_str)
    return transformed_path
    
#
def loc_str(target_str='\\\\',orignal_str):
    # in windows,you need change the string of '\' in to '\\\\'
    loc_start=[]
    loc_end=[]
    s=re.finditer(target_str,orignal_str)
    for i in s:
        loc_start.append(i.start())
        loc_end.append(i.end())
    return loc_start,loc_end
   
