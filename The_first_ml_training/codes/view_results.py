# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:54:53 2020

@author: Li Chao
Email: lichao19870617@163.com
"""

import pickle

output_file = "./outputs.pickle"
stat_file = "./stat.pickle"

with open(output_file, 'rb') as f:
    output = pickle.load(f)

with open(stat_file, 'rb') as f:
    stat = pickle.load(f)