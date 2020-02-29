# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:27:40 2018

@author: lenovo
"""

# import
 	
#from bs4 import BeautifulSoup
import numpy as np
import requests,re,time
import pandas as pd
#
# input
#expr_title=r'nossl=1">([^<].*?)</a>'
def extract_allUrl(pages,expr_title,expr_citeTime):
    for page in pages:
        time.sleep(1)
        print('page {}'.format(page))
        url=gen_Url(page)
        try:
            info=extract_oneUrl(url,expr_title,expr_citeTime)
            save_to_excel(info)
        except ValueError:
            print('url {} have book, please check it'.format(url))
    return 
    
def extract_oneUrl(url,expr_title,expr_citeTime):
    # cite times
    html = getHTMLText(url)
    citeTime=re.findall(expr_citeTime,html)
    citeTime = [eval(x.split('<')[0][6:]) for x in citeTime ]
    # title
#    expr_title=r'nossl=."(.*?)</a>'
    title= re.findall(expr_title,html)
#   delete others
    n_title=len(title)
    index=[]
    for i in range(n_title):
        if re.findall('><span',title[i])==['><span']:
            index.append(i)
    title=pd.DataFrame(title)
    title=title.iloc[list(set(title.index)^set(index))]
    title=list(title[0])
    title=[re.sub(r'<b>','',ti) for ti in title]
    title=[re.sub(r'</b>','',ti) for ti in title]
    title=[re.sub(r'>','',ti) for ti in title]
    #
    info=np.vstack([np.array(title),np.array(citeTime)])
    info=np.transpose(info)
    info=pd.DataFrame(info)
    return info


def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
#
def save_to_excel(info):
    writer = pd.ExcelWriter('Save_Excel.xlsx')
    info.to_excel(writer,'sheet1',float_format='%.5f') # float_format 控制精度
    writer.save()
    
#    
def gen_Url(page):
    url='https://b.beijingbang.top/scholar?start='+\
    str(page)+\
    '1&q=schizophrenia+and++fmri&hl=zh-CN&as_sdt=0,5'    
    return url
    
if __name__=='__main__':
#    url='https://b.beijingbang.top/scholar?q=schizophrenia+and++fmri'
    url='https://b.beijingbang.top/scholar?start=100&q=schizophrenia+and++fmri&hl=zh-CN&as_sdt=0,5'
    url='https://b.beijingbang.top/scholar?start=1&q=schizophrenia+and++fmri&hl=zh-CN&as_sdt=0,5'
    expr_title_article=r'nossl=."(.*?)</a>'
    expr_title_book=r'\[引用\]</span>.*?</h3>'
    expr_title=r'nossl=."(.*?)</a> | \[引用\]</span>.*?</h3>'
#    title= re.findall(expr_title_article,html)
    expr_citeTime=r'被引用次数.*?:.*?\d*</a> <a'
    pages=[9,10]
    info=extract_allUrl(pages,expr_title,expr_citeTime)