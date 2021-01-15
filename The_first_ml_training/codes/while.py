# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:35:56 2020

@author: lenovo
"""


# i = 0
# while i < 3:
#     print(string[i])
#     # print(i)
#     i = i + 1

# i = 0
# while i < 10:
#     print(i)
#     i = i + 1  # i += 1
    
string = ["I", "love", "python", 1,2,3,4,1,2,3,4,1,2,3,1]
s = "none"
i = 0

count = 0
idx = []
while (s != 1) or (count < 4):
    s = string[i]
    
    if s == 1:
        count = count+1
        idx.append(i)

    i = i + 1
  
    