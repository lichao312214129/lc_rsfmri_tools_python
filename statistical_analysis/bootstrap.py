# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:21:10 2020

@author: lenovo
"""

import numpy as np


def average(data):
    return sum(data) / len(data)


def bootstrap(data, B, c, func):
    """
    计算bootstrap置信区间
    :param data: array 保存样本数据
    :param B: 抽样次数 通常B>=1000
    :param c: 置信水平
    :param func: 计算样本
    :return: bootstrap置信区间上下限
    """
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return lower, higher


if __name__ == '__main__':
    from kappa import quadratic_weighted_kappa
    import pandas as pd
    file = r'D:\workstation_b\limengsi\加权Kappa.xlsx'
    data = pd.read_excel(file, sheet_name="2D")
    result = bootstrap(data, 1000, 0.95, quadratic_weighted_kappa)
    print(result)