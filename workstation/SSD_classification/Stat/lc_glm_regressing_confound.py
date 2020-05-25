# -*- coding: utf-8 -*-
"""This script is used to regressing confounds: age, sex, headmotion and site.
"""

import pandas as pd
import numpy as np

data = np.load(r'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_206.npy')