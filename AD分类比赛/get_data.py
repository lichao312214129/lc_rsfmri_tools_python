# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:57:52 2020
Split the data into training, validation and test sets
@author: Li Chao, Dong Mengshi
"""

from sklearn.model_selection  import StratifiedShuffleSplit
import scipy.io as sio
import numpy as np
from preprocess import preprocess
import pandas as pd
import matplotlib.pyplot as plt

def get_data(seed=66):
    # Input
    file = r'F:\AD分类比赛\MCAD_AFQ_competition.mat'

    # Load
    data_struct = sio.loadmat(file)  
    label = data_struct["train_diagnose"]
    site = data_struct["train_sites"]
    data_for_all, label = preprocess(file)

    # Get training , validation and test sets
    data_for_all = {key: data_for_all[key] for key in data_for_all.keys()}

    return data_for_all, label


