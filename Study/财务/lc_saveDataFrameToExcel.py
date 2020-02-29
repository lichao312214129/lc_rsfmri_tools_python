# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:36:14 2018
save one DataFrame to excel
@author: lenovo
"""
# import 
import pandas as pd
# def
def saveOneToExcel(content,excelName='output.xlsx'):
#    Content[0].to_csv('Result.csv',\
#           sep='|',index=1,na_rep='NA',float_format='%.2f',\
#           encoding='utf-8')
#    startrow=0
#    startcol=0 
    writer = pd.ExcelWriter(excelName)
    content.to_excel(writer,'Sheet1',float_format='%20.2f',\
                     index=False)
    writer.save()