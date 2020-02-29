# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:02:22 2018

@author: lenovo
"""
# import
import requests
import re
#
#url='http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/300027.phtml?year=2018&jidu=1'
expr_date=r'\s+(\d\d\d\d-\d\d-\d\d)\s*'
expr_s=r'<td><div align="center">(\d*\.?\d*)</div></td>'
expr_LowPlusTotle=r'<td class="tdr"><div align="center">(\d*\.?\d*)</div></td>'
url='https://b.beijingbang.top/scholar?q=schizophrenia+and++fmri'
expr=r'noss=1.*?</a>*'

nossl=1">Neurobiology of smooth pursuit eye movement deficits in <b>schizophrenia</b>: an <b>fMRI </b>study</a>
#exp_cash='<div align="center">(\d*|\d*\.?\d*\d*)</div>'
#cash = re.findall(exp_cash,html)
#date=re.findall(exp_date,html)
#cash[0].split('>' and '<')
#re.findall(r'^\d','anncbd3456.889123455')

###########

 
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
    
def parsePage(ilt, html,expr):
    try:
        date=re.findall(expr_date,html)
        con_s= re.findall(expr_s,html)
        con_LowPlusTotle=re.findall(expr_low,html)
    except:
        print("")
        
def main():
    infoList = []
    html = getHTMLText(url)
    parsePage(infoList, html)
    return infoList

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(1,60,1)
b=[int(xx) for xx in b]
plt.plot(x,b)