# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:44:59 2018
google scholar 
@author: lenovo
"""

# import
 	
from bs4 import BeautifulSoup
import numpy as np
import requests,time,re,bs4
# input

# def
def extractTitle(url):
    html = getHTMLText(url)
    soup = BeautifulSoup(html,'html.parser')
    html=soup.prettify()
    ulist=[]
    http=[]
    for content in soup.find_all('dd'):
        if isinstance(content,bs4.element.Tag):
            title=content('li')
            allString=content.get_text()
            allString=allString.split('\n')
            http.append(content.a.get('href'))
            ulist.append(allString)
    return ulist,http

def extractTimes(soup):
#    expr_citeTimes=r'被引用次'
    citeTimes=soup.find_all('a')
    citeTimes= [eval(x.split('<')[0][6:]) for x in citeTimes ]
    return citeTimes
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
        
#if __name__=='__main__':
#    url='https://www.jobmd.cn/work/Medical_imaging_radiology.htm'
#    expr_title=r'nossl=1">([^<].*?)</a>'
#    info=extract_onePage(url,expr_title)
        url=r'https://isisn.nsfc.gov.cn/egrantindex/funcindex/prjsearch-list'