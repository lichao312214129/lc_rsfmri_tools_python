# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import chardet


class ConcatTable():
    """ Extract sub-data from multiple table according with unique id in column of header_unique_id, 
    then concatenate multiple table to one, axis=0. 

    NOTE: This class will iterate all unique id
    
    Parameters
    ----------
    root : directory, default None
        All files (e.g., csv, excel) in the directory.

    header_unique_id: string, default None
        Which header contains the unique index, unique index could be any content, e.g., bank account, ID.

    header_username: string, default None
        Which header contains the username that used to generate the save name suffix, e.g., name.

    savedir: path, Default None
        Which directory to save the results

    savename: string, default None
        save name behind the suffix of header_username

    encoding: string, default 'gb18030'
        encoding of data
    """


    def __init__(self, root=None, header_unique_id=None, header_username=None, 
        savedir=None, encoding='gb18030'):
        self.root = root
        self.header_unique_id = header_unique_id
        self.header_username = header_username
        self.savedir = savedir
        self.encoding = encoding
        
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def identify_unique_num(self):
        """Identify unique numbers of one header_unique_id (pd.Series) in several files under the same directory.
        """

        file = os.listdir(self.root)
        self.file_path = [os.path.join(self.root, file_) for file_ in file]
        
        # # Read table
        # f = open(self.file_path[0],'rb') 
        # data = f.read() 
        # file_encoding = chardet.detect(data).get('encoding')
        uni_num = []
        for i, file_ in enumerate(self.file_path):
            data = pd.read_csv(file_, encoding=self.encoding)
            num = data[self.header_unique_id].unique()
            uni_num.extend(num)
        
        # uni_num to str
        uni_num = [str(num) for num in uni_num]
        self.uni_num = np.unique(uni_num)
            
        return self
    
    def concat_for_all_num(self):
        for num in self.uni_num:
            print(f"Combining for {num}...\n")
            self.concat_for_one_account(num)
    
    def concat_for_one_account(self, num):       
        data_concat = pd.DataFrame([])
        for i, file_ in enumerate(self.file_path):
            data = pd.read_csv(file_, encoding=self.encoding)
            savename = data[self.header_username]
            dh = data[self.header_unique_id].copy()
            dh =  pd.Series([str(dh_) for dh_ in dh])
            loc = dh.isin([num])
            data_ = data[loc]
            if np.sum(loc):
                savename = savename[loc].iloc[0]
                savename_ = savename
            else:
                savename = ""
            print(f"\tConcat for {file_}\length={len(data_)}\n")
            data_concat = pd.concat([data_concat,data_])
            
        # save
        data_concat[self.header_unique_id] = [str(nu) + '\t' for nu in data_concat[self.header_unique_id]]
        data_concat.to_csv(os.path.join(self.savedir, savename_ + str(num) + ".csv"), index=False, encoding=self.encoding)


# ===============================以下需要修改==================================
if __name__ == '__main__':
    concat = ConcatTable(
        root = r'D:\workstation_b\finacial\孙敏等',  # 需要整合的文件所在的文件夹
        header_unique_id = "交易卡号",  # 整理的列名
        header_username = "客户名称",  # 客户名称所在的列名
        savedir = r'D:\workstation_b\finacial\孙敏各个账户',  # 保存到哪个文件夹
    )
       
    concat.identify_unique_num()
    concat.concat_for_all_num()
    print("Done!\n")
    input()