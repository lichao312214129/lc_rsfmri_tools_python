# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:27:24 2020

@author: lenovo
"""

import pandas as pd
import numpy as np
import os
import re
import chardet
    
    
class ExtractInfoForFinacial():
    def __init__(self, 
                source_file=None, 
                user_name=None, 
                header_transaction_cardnumber=None,
                header_target_name=None,
                header_target_cardnumber=None,
                header_income_pay_label=None,
                label_income=None,
                label_pay=None,
                header_transaction_amount=None,
                header_transaction_notes=None,
                save_path=None
        ):

        # ----------------------------------------------------------------
        self.source_file = r'D:\workstation_b\finacial\job001\流水\昱盛0902.csv'
        self.user_name = "昱盛0902"
        
        self.header_transaction_cardnumber = "卡号"
        self.header_target_name = "对方户名"
        self.header_target_cardnumber = "对方帐户"
        self.header_income_pay_label = "借贷标志"
        self.label_income = '进'
        self.label_pay = '出'
        self.header_transaction_amount = "发生额"
        self.header_transaction_notes = "注释"
        self.save_path = r'D:\workstation_b\finacial\job001'
        # ----------------------------------------------------------------
    
    def load_data(self):
        # Get encoding
        f = open(self.source_file,'rb') 
        data = f.read() 
        file_encoding = chardet.detect(data).get('encoding')
        
        # Load data
        load_method = {"xlsx": pd.read_excel, "xls": pd.read_excel, "csv": pd.read_csv}  # TODO: extending to other file formats
        suffix = self.source_file.split('.')[-1]
        try:
            self.source = load_method.get(suffix)(self.source_file, encoding=file_encoding)
        except UnicodeDecodeError:
            self.source = load_method.get(suffix)(self.source_file, encoding="gbk")
        
        self.source = self.source.replace(np.nan, "")
        self.source = self.source.replace("nan", "")
        return self
    
    def get_standard_header_name(self):
        """
        将表头标准化
        
        通过正则匹配的方式，找到对应到标准表头的列的位置，然后将改位置的表头名改为标准表头名
        TODO: 此处不可能穷尽所有的情况，所以在实践工作中需要不断的完善正则匹配
        """
        
        # Del space in columns
        columns = list(self.source.columns)
        columns = [col.strip() for col in columns]
        self.source.columns = columns

        # 找到匹配的列的位置
        columns = pd.Series(self.source.columns)
        loc_账号 = columns.str.match(r"(?!.*对[方手]).*卡号.*")
        loc_income_pay_label = columns.str.match(r".*(借贷标志|收付标志|进出|交易方向|交易类型).*")
        loc_transaction_amount = columns.str.match(r".*(金额|发生额).*")
        loc_target_name = columns.str.match(r".*对[方手].*(户名|名称).*")
        loc_对方账号 = columns.str.match(r".*对[方手].*[账帐卡][户号].*")
        loc_transaction_notes = columns.str.match(r".*(注释|备注|摘要).*")
        
        columns.str.match(".*对手.*(户名|名称).*")
        
        # 将匹配的列的表头名改为标准表头名
        self.source = self.revise_header_to_standard_header(self.source, loc_账号, self.header_transaction_cardnumber)
        self.source = self.revise_header_to_standard_header(self.source, loc_income_pay_label, self.header_income_pay_label)
        self.source = self.revise_header_to_standard_header(self.source, loc_transaction_amount, self.header_transaction_amount)
        self.source = self.revise_header_to_standard_header(self.source, loc_target_name, self.header_target_name)
        self.source = self.revise_header_to_standard_header(self.source, loc_对方账号, self.header_target_cardnumber)
        
        self.source = self.revise_header_to_standard_header_notes(self.source, loc_transaction_notes, self.header_transaction_notes)
        
        # Del space in self.header_income_pay_label
        self.source[self.header_income_pay_label] = [hipl.strip() for hipl in self.source[self.header_income_pay_label]]
        
        # Replace income_pay_label to self.label_pay, self.label_income
        loc_pay = self.source[self.header_income_pay_label].str.contains(r'借|出|付|D|取现|现取|提现|消费|0011|0001|^[0]$|退')
        self.source[self.header_income_pay_label][loc_pay] = self.label_pay
        self.source[self.header_income_pay_label][loc_pay.isin([False])] = self.label_income
    
    @ staticmethod
    def revise_header_to_standard_header(df, loc, loc_name):
        if len(df.columns[loc]) > 1:
            raise RuntimeError(f"存在多个待修改为标准表头'{loc_name}'的非标准表头\n这些非标准表头为: {df.columns[loc]}")   
        if len(df.columns[loc]) == 0:
            raise RuntimeError(f"未匹配到待修改为标准表头'{loc_name}'的非标准表头\n请添加相应的正则表达式")   
        else:
            columns = np.array(df.columns)
            columns[loc] = loc_name
            df.columns = columns
        return df
    
    @ staticmethod
    def revise_header_to_standard_header_notes(df, loc, loc_name): 
        """
        将多个注释融合成一列，并将表头命名为 loc_name
        """
        
        if len(df.columns[loc]) == 0:
            raise RuntimeError(f"未匹配到待修改为标准表头'{loc_name}'的非标准表头\n请添加相应的正则表达式")   
        else:
            columns = np.array(df.columns)
            data_notes = df[columns[loc]]
            data_notes = pd.DataFrame(data_notes, dtype=str)
            note = ""
            for hdn in data_notes.columns:
                data_notes[hdn] = data_notes[hdn].replace(np.nan, "")
                data_notes[hdn] = data_notes[hdn].replace("nan", "")
                note = note + "--" + data_notes[hdn]
                
            df[loc_name] = note
            
        return df

    def extract(self):
        unique_target_cardnumber = self.source[self.header_target_cardnumber].unique()
        self.extracted_data = pd.DataFrame()
        for ith, utc in enumerate(unique_target_cardnumber):
            loc = self.source[self.header_target_cardnumber].isin([utc])
            loc_income = loc & (self.source[self.header_income_pay_label] == self.label_income)
            loc_pay = loc & (self.source[self.header_income_pay_label] == self.label_pay)
            
            transaction_cardnumber = str(self.source[self.header_transaction_cardnumber][loc].iloc[0])  # TODO: one person have several card number
            target_name = self.source[self.header_target_name][loc].iloc[0].strip()
            target_account = self.source[self.header_target_cardnumber][loc].iloc[0]
            
            income = self.source[self.header_transaction_amount][loc_income]
            income_note = self.source[self.header_transaction_notes][loc & (self.source[self.header_income_pay_label] == self.label_income)]
            income_note_sort_according_income = self.sort_note_according_money(income, income_note)
            income_sum = income.sum()
            
            pay = self.source[self.header_transaction_amount][loc_pay]
            pay_note = self.source[self.header_transaction_notes][loc_pay]
            pay_note_sort_according_pay = self.sort_note_according_money(pay, pay_note)
            pay_sum = pay.sum()
            
            if pay_sum != 0:
                net_disbursement = pay_sum - income_sum
            else:
                net_disbursement = None
            
            # Combine
            extracted_data_ = pd.DataFrame({
                "户名": self.user_name, 
                "账号": transaction_cardnumber, 
                "支出": pay_sum, 
                "收入": income_sum, 
                "净支出": net_disbursement, 
                "对方户名": target_name, 
                "对方账号": target_account, 
                "收入备注": income_note_sort_according_income, 
                "支出备注": pay_note_sort_according_pay}, index=[ith])
            
            self.extracted_data  = pd.concat([self.extracted_data, extracted_data_])
            
    def save(self):
        self.extracted_data["账号"].astype('str')
        self.extracted_data.to_excel(os.path.join(self.save_path, self.user_name + "报告.xlsx"),  encoding='GB2312')
    
    @staticmethod
    def sort_note_according_money(money, note):
        uni_note = note.unique()
        note_money = []
        for un in uni_note:
            loc_note = note.isin([un])
            note_money.append(un.strip() + ": " + str(money[loc_note].abs().sum()) + "; ")
        
        money_order = [nm.split(";")[0].split(":")[-1] for nm in note_money]
        money_order = [eval(mo) if isinstance(mo, str) else 0 for mo in money_order]
        order = np.argsort(money_order)[::-1]
        note_money = np.array(note_money)[order]
        return "".join(note_money)


if __name__ == '__main__':
    extractor = ExtractInfoForFinacial()
    extractor.load_data()
    extractor.get_standard_header_name()
    extractor.extract()
    extractor.save()