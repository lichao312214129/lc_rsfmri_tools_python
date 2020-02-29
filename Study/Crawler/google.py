# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:02:22 2018
google scholar
@author: lenovo
"""
# import
 	
from bs4 import BeautifulSoup
import numpy as np
import requests
import re
#
# input
#expr_title=r'nossl=1">([^<].*?)</a>'
def extract_onePage(url,expr_title):
    expr_citeTime=r'被引用次数.*?:.*?\d*</a> <a'
    html = getHTMLText(url)
    title= re.findall(expr_title,html)
    title=[re.sub(r'<b>','',str) for str in title ]
    title=[re.sub(r'</b>','',str) for str in title ]
    citeTime=re.findall(expr_citeTime,html)
    citeTime = [eval(x.split('<')[0][6:]) for x in citeTime ]
    info=np.vstack([np.array(title),np.array(citeTime)])
    info=np.transpose(info)
    return info
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
        
if __name__=='__main__':
#    url='https://b.beijingbang.top/scholar?q=schizophrenia+and++fmri'
    url='https://b.beijingbang.top/scholar?start=100&q=schizophrenia+and++fmri&hl=zh-CN&as_sdt=0,5'
    expr_title=r'nossl=1">([^<].*?)</a>'
    info=extract_onePage(url,expr_title)