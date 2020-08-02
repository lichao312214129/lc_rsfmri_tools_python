# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:26:32 2020

@author: lenovo
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import os


class MergeTable():
    """Merge multiple tables with one table to multiple tables (multiple VS. one).

    Parameters
    ----------
    root_dir_multiple_files: directory
        root directory of the 'multiple tables'

    one_file: file path
        file path of the 'one table'

    header_multiple_files: string
        header of 'multiple tables' that used for merging

    header_one_file: string
        header of the 'one table' that used for merging

    encoding: string, default 'gb18030'
    """

    def __init__(self, 
                 root_dir_multiple_files=None, one_file=None, 
                 header_multiple_files=None,  header_one_file=None, 
                 savedir=None, savename=None, encoding="gb18030"):

        self.root_dir_multiple_files = root_dir_multiple_files
        self.one_file = one_file
        self.header_multiple_files = header_multiple_files
        self.header_one_file = header_one_file
        self.savedir = savedir
        self.encoding = encoding
    
    def get_all_files(self):
        self.multiple_files = [os.path.join(self.root_dir_multiple_files, name) for name in os.listdir(self.root_dir_multiple_files)]
        return self

    def merge_all(self):
        data2 = pd.read_excel(self.one_file, encoding=self.encoding, sep=',', dtype=str)
        for file1 in self.multiple_files:
            print(f"Merge table for {file1}\n")
            data1 = pd.read_csv(file1, encoding=self.encoding, sep=',', dtype=str)
            self.merge(data1,data2, file1.split(".")[0] + '与被害人财务往来.csv')
        
    def merge(self, data1, data2, savename):
        # data to str for matching
        data1[self.header_multiple_files] = [str(dh).strip()  for dh in data1[self.header_multiple_files]]
        data2[self.header_one_file] = [str(dh).strip()  for dh in data2[self.header_one_file]]
        data_merge_inner = pd.merge(data1, data2, left_on=self.header_multiple_files, right_on=self.header_one_file, how='inner')
        
        if len(data_merge_inner) != 0:
            # To str
            for h in data_merge_inner.columns:
                data_merge_inner[h] = [str(dh).strip() + '\t' for dh in data_merge_inner[h]]
            
            data_merge_inner.to_csv(os.path.join(self.savedir, savename), sep=',', encoding=self.encoding)
        


# ===============================以下需要修改==================================
if __name__ == "__main__":
    merge = MergeTable(
        root_dir_multiple_files = r'D:\workstation_b\finacial\孙敏各个账户',  # 嫌疑犯文件所在文件夹
        one_file = r'D:\workstation_b\finacial\名单03.xlsx',  # 受害人的文件
        header_multiple_files = '对手帐号',  # 嫌疑犯的融合表头
        header_one_file = '银行卡号',  # 被害人的融合表头
        savedir = r'D:\workstation_b\finacial\孙敏各个账户',  # 保存到哪里
        # -----
        encoding="gb18030")

    merge.get_all_files()
    merge.merge_all()
    print("Done!\n")
    input()