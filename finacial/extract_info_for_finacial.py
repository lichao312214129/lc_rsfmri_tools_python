# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:27:24 2020

@author: lenovo
"""

import pandas as pd
import numpy as np
import os
    
    
class ExtractInfoForFinacial():
    def __init__(self):
        # ----------------------------------------------------------------
        self.source_file = r'D:\workstation_b\finacial\job001\流水\邹海洋-工商4198.xls'
        self.user_name = "邹海洋"
        self.header_transaction_cardnumber = "卡号"
        self.header_target_cardnumber = "对方帐户"
        self.header_target_name = "对方户名"
        self.header_income_pay_label = "借贷标志"
        self.label_income = '贷'
        self.label_pay = '借'
        self.header_transaction_amount = "发生额"
        self.header_target_transaction_notes = "注释"
        # ----------------------------------------------------------------
    
    def load_data(self):
        # Load
        if os.path.basename(self.source_file).split('.')[-1] in "xlsx":
            self.source = pd.read_excel(self.source_file, encoding='GB2312')
        else:
            self.source = pd.read_csv(self.source_file, encoding='GB2312')

        return self
    
    def preprocessing(self):
        # Del space in columns
        columns = list(self.source.columns)
        columns = [col.strip() for col in columns]
        self.source.columns = columns
        
        # Del space in self.header_income_pay_label
        self.source[self.header_income_pay_label] = [hipl.strip() for hipl in self.source[self.header_income_pay_label]]

        return self
    
    def extract(self):
        unique_target_cardnumber = self.source[self.header_target_cardnumber].unique()
        self.extracted_data = pd.DataFrame()
        for ith, utc in enumerate(unique_target_cardnumber):
            loc = self.source[self.header_target_cardnumber].isin([utc])
            loc_income = loc & (self.source[self.header_income_pay_label] == self.label_income)
            loc_pay = loc & (self.source[self.header_income_pay_label] == self.label_pay)
            
            transaction_cardnumber = " " + str(self.source[self.header_transaction_cardnumber][loc].iloc[0])  # TODO: one person have several card number
            target_name = self.source[self.header_target_name][loc].iloc[0].strip()
            target_account = self.source[self.header_target_cardnumber][loc].iloc[0]
            
            income = self.source[self.header_transaction_amount][loc_income]
            income_note = self.source[self.header_target_transaction_notes][loc & (self.source[self.header_income_pay_label] == self.label_income)]
            income_note_sort_according_income = self.sort_note_according_money(income, income_note)
            income_sum = income.sum()
            
            pay = self.source[self.header_transaction_amount][loc_pay]
            pay_note = self.source[self.header_target_transaction_notes][loc_pay]
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
        self.extracted_data.to_excel("extracted_data.xlsx",  encoding='GB2312')
    
    @staticmethod
    def sort_note_according_money(money, note):
        uni_note = note.unique()
        note_money = []
        for un in uni_note:
            loc_note = note.isin([un])
            note_money.append(un.strip() + ": " + str(money[loc_note].abs().sum()) + "; ")
        
        money_order = [nm.split(";")[0].split(":")[1] for nm in note_money]
        money_order = [eval(mo) if isinstance(mo, str) else 0 for mo in money_order]
        order = np.argsort(money_order)[::-1]
        note_money = np.array(note_money)[order]
        return "".join(note_money)


if __name__ == '__main__':
    extractor = ExtractInfoForFinacial()
    extractor.load_data()
    extractor.preprocessing()
    extractor.extract()
    extractor.save()