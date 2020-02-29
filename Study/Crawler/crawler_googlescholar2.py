# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 18:41:19 2018
谷歌学术爬虫:2014年以后，keywords=
@author: lenovo
"""

from lxml import etree
import pandas as pd
import requests,time,re
import numpy as np
#
#Some User Agents
hds=[{'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'},\
    {'User-Agent':'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'},\
    {'User-Agent':'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'},\
    {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0'},\
    {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/44.0.2403.89 Chrome/44.0.2403.89 Safari/537.36'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
    {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
    {'User-Agent':'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
    {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11'},\
    {'User-Agent':'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11'},\
    {'User-Agent':'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11'}]
# def
def gen_Url(page,url,keyword,year):
    url=url+str(page)+'&q='+keyword+'&hl=zh-CN&as_sdt=0,5&as_ylo='+str(year) 
    return url  
#
def getHTMLText(url):
    try:
        r = requests.get(url, headers=hds[np.random.randint(0,len(hds)-1)],timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""

def parse(html):
    browser = etree.HTML(html)
    # title
    title=browser.xpath("//div[@id='gs_res_ccl_mid']/div/div[@class='gs_ri']/h3/a")
    title=[str(title_.xpath('string(.)')) for title_ in title]
    
    http=browser.xpath("//h3/a[@href]")
    http=[str(http_.xpath('@href')) for http_ in http]
    
    citeTimes=browser.xpath("//div[@class='gs_fl']")
    citeTimes=[str(citeTimes_.xpath('string(.)')) for citeTimes_ in citeTimes]
    citeTimes=[re.findall('([1-9]*\d*)',citeTimes_.split(' ')[2]) for citeTimes_ in citeTimes]
    #去掉空字符串
    Cite=[]
    for citetimes in citeTimes:
        cite=[citeTimes_ for citeTimes_ in citetimes if citeTimes_ is not ''  ]
        Cite.append(cite)
    citeTimes=Cite
    # years
    pub_year=browser.xpath("//div[@class='gs_a']")
    pub_year=[str(pub_year_.xpath('string(.)')) for pub_year_ in pub_year]
    pub_year=[re.findall('[1-9]*\d*',pub_year_) for pub_year_ in pub_year]
    Year=[]
    for year in pub_year:
        puby=[year_ for year_ in year if year_ is not ''  ]
        Year.append(puby[0])
    pub_year=Year
    #
    allList=[title,citeTimes,pub_year,http]
    allinfo=[pd_transf(mylist) for mylist in allList]
    allInfo=pd.concat(allinfo,axis=1)
    allInfo.columns=['title','cite_times','year','url']
    return allInfo
    
def pd_transf(mylist):
    return pd.DataFrame(mylist)
#===========================================================
if __name__=='__main__':
    n_pages=20
    seed_url='https://a.glgoo.top/scholar?start='
    Url=[gen_Url(i*10,seed_url,keyword='schizophrenia+AND+fmri',year=2014) for i in range(n_pages)]
    #
    allInfo=pd.DataFrame([],columns=['title','cite_times','year','url'])
    for i,url in enumerate(Url):
        print('fetch page {}/{}'.format(i+1,n_pages))
        html=getHTMLText(url)
        time.sleep(np.random.randint(2,5))
        if html.strip():
            allinfo=parse(html)
            allInfo=pd.concat([allInfo,allinfo],axis=0)
    
    # save
    allInfo=allInfo.where(allInfo.notnull(),'0')
    allInfo[['cite_times','year']] = allInfo[['cite_times','year']].apply(pd.to_numeric)
    allInfo=allInfo.sort_values('cite_times',ascending=False)
    allInfo.to_excel('googlescholar.xlsx',index=False,header=True)
    allInfo.to_csv('googlescholar.txt',index=False,header=True)
    print('All done!')
        