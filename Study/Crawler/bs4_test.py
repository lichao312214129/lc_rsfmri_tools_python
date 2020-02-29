# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:59:47 2018

@author: lenovo
"""
from bs4 import BeautifulSoup
import bs4
import numpy as np
import requests
import re
def main():
    url='http://www.zuihaodaxue.cn/keyanguimopaiming2018.html'
    html=getHTMLText(url)
    soup=getSoup(html)
    ulist=[]
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds=tr('td')
#            print(tds[1].string)
            ulist.append([tds[0].string,tds[1].string,tds[3].string])
        
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
def getSoup(html):
    soup = BeautifulSoup(html,'html.parser')
    html=soup.prettify()
    return soup