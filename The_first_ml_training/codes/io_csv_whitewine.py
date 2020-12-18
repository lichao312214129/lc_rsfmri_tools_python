# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:13:18 2020

@author: lenovo
"""

# 导入模块
import pandas as pd
import numpy as np

# 加载数据
data_file = '../demo_data/winequality-white.csv'
data = pd.read_csv(data_file, sep=';')

# 查看数据
describe = data.describe()
data.info()

# 给data加一列ID，方便后续图形界面操作
data["__ID__"] = np.arange(0, len(data))
data = data.iloc[:,::-1]  # 倒序，让ID放在第一列

# 将"质量"改为__Targets__,方便后续图形界面操作
data.rename(columns={"quality":"__Targets__"}, inplace=True)

# 获取数据的特征和targets
features = data.iloc[:, np.hstack([0, np.arange(2,13)])]
targets = data.iloc[:, [0,1]]  

# 查看有几个类别
np.unique(targets["__Targets__"])

# 我们把3,4设置为第一个类别，5，6设置为第二个类别，7，8设置为第三个类别
targets.loc[targets["__Targets__"].isin([3, 4, 5]), "__Targets__"] = 0 
targets.loc[targets["__Targets__"].isin([6, 7,8,9]), "__Targets__"] = 1


# 再次查看label
np.sum(targets["__Targets__"]==0)
np.sum(targets["__Targets__"]==1)
np.unique(targets["__Targets__"])

# 保存到excel
features.to_csv("../demo_data/features_whitewine.csv", index=False)
targets.to_csv("../demo_data/targets_whitewine.csv", index=False)
