# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:35:57 2020

@author: Li Chao
Email: lichao19870617@163.com
"""

class Dog():
    def __init__(self, skin="白色", length=0.1):
        print("实例化了")
        self.skin = skin
        self.length = length
        self.ran = "No"
        self.wanged = "No"

    def run(self, km):
        print(f"跑了{km}千米")
        self.ran = "Yes"

    def wang(self):
        print("叫。。。")  
        self.wanged = "Yes" 
        
 
dog = Dog(skin="黑色", length=0.9)  # 实例化Dog类, 用dog来表示Dog类

# ran = "紫色"
# dog = Dog(ran)

# print(dog.ran)
# print(dog.skin)
# dog.run(km=100)
# print(dog.ran)


