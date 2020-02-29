# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:53:36 2018

@author: lenovo
"""

# import
 	
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests,time,bs4,re
from sklearn.externals.joblib import Parallel, delayed
# input
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
def gen_Url(page,url_profession):
    url=url_profession+str(page)    
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
#
def extract_onePage(url):
    html = getHTMLText(url)
    soup = BeautifulSoup(html,'html.parser')
    html=soup.prettify()
    ulist=[]
    http=[]
    for content in soup.find_all('dd'):
        if isinstance(content,bs4.element.Tag):
#            title=content('li')
            allString=content.get_text()
            allString=allString.split('\n')
            http.append(content.a.get('href'))
            ulist.append(allString)
    return ulist,http

def extract_allPage(url):
#    ulist,http=[extract_onePage(u) for u in url]
    Info=[]
    Http=[]
    for i,u in  enumerate(url):  
        time_sleep=np.random.randint(1,4)
        print('page{}\tsleep time is {}'.format(i+1,time_sleep))
        time.sleep(time_sleep)
        ulist,http=extract_onePage(u)
        Info.append(ulist)
        Http.append(http)
    return Info,Http

def screen_info(http,Info):
    # 筛选
    Http=[http_temp[15:35] for http_temp in http]
    info=[]
    for page in Info:
        page_job_2d=page[15:35]
        #除掉空列表
        content=[]
        for page_one in page_job_2d:
            content.append([page_temp for page_temp in page_one if page_temp !=''])
            Content=[c[0:3] for c in content]
        info.append(Content)
    return Http,info   
 
def fetch_detail_one(http):    
    # 把每个http的内容提取出来
    html=getHTMLText(http)
    soup = BeautifulSoup(html,'html.parser')
    html=soup.prettify()
    detail_1=soup.findAll('div',{'class':'box-info_base'})
    detail_2=soup.findAll('div',{'class':'box-info_other'})
    detail_all=soup.findAll('dl',{'class':'work-group'})

#    try:
    n_detail_all=len(detail_all)-2
    detail_describe=[]
    for i in range(n_detail_all):
        cmd="soup.findAll('dl',{'class':'work-group'})["+\
         str(i)+"]"
        detail_describe.append(eval(cmd))
#    except:
#        print('no description')
    # 筛选详细信息
    # 1
    try:
        tag_salary=detail_1[0]
        detail_1=tag_salary.findAll('span')
        detail_1=[detail_1_.string for detail_1_ in detail_1]
    except IndexError:
        detail_1=['None','None','None','None','None']

    # 2
    try:
        tag_salary=detail_2[0]
        detail_2=tag_salary.findAll('span')
        detail_2=[detail_2_.string for detail_2_ in detail_2]
    except IndexError:
        detail_2='None'
            
    # 3
    detail_describe_=[[],[],[]]
    rege=re.compile(r"<dd>|\\r|\\n|\\\s|</dd>|r|\n|b/|<span>|</span>|<a>|</a>|<b>|</b>|<b/>|"+\
                    '<dd class="clearfix">|<ul class="work-require">|<li>|</li>|<label>|</label>|</p>|<|>|" "')
    for i in range(n_detail_all):
        cmd="detail_describe["+str(i)+"].findAll('dd')[0]"     
        temp=eval(cmd)
        temp=str(temp)
        temp = re.sub(rege, '', temp)
        if n_detail_all==3:
            detail_describe_[i]=temp
        elif n_detail_all==2:
            detail_describe_[i+1]=temp
    # concat
    # 总结
    try:
        salary,years,edu,fullORpart=detail_1[0],detail_1[2],\
                                    detail_1[3],detail_1[4]
    except IndexError:
        salary,years,edu,fullORpart='None','None','None','None'
                                        
    duration=str(detail_2)
        
    detail_list=[salary,years,edu,fullORpart,duration,\
                 detail_describe_[0],detail_describe_[1],detail_describe_[2]]
    
    return detail_list

def fetch_detail_all(Http): 
    detail_all=[]
    for count,http_list in enumerate(Http):
        print('fetching detail of page {}'.format(count+1))
        detail_list=[fetch_detail_one(http_) for http_ in http_list ] 
        detail_all.append(detail_list)
        # save
#        detail_all_pd=pd.DataFrame(detail_all)
#        save(detail_all_pd,str(count)+'.xlsx')
    return detail_all

#def fetch_detail_all_multiprocess(Http):
#    # 多线程
#    s=time.time()
#    # 当复制的文件较少时，不要开多线程
#    if len(Http)<=20:
#        n_processess=1
#        
#    print('Copying...\n')
#    Parallel(n_jobs=n_processess,backend='threading')\
#        (delayed(fetch_detail_one)(http_)\
#         for http_ in http_list)
#    e=time.time()
#    print('Done!\n running time is {:.1f}'.format(e-s))

def concat_allInfo(Http,info,df_detail):
#     组合信息到pandas格式   
    df_1 = [pd.DataFrame(info_,\
                        columns=['职位','单位','地区']) for info_ in info]
    df_http = [pd.DataFrame(Http_,\
                    columns=['链接']) for Http_ in Http]
    df_detail = [pd.DataFrame(df_detail_,columns=\
                ['薪水','工作年限','学历','是否全职','招聘时间','职位亮点','职位描述','职位要求']) for df_detail_ in df_detail]
    df_1=pd.concat(df_1,axis=0)
    df_http=pd.concat(df_http,axis=0)
    df_detail=pd.concat(df_detail,axis=0)
    df=pd.concat([df_1,df_http,df_detail],axis=1)
    return df

def save(df,name):
    writer = pd.ExcelWriter(name)
    df.to_excel(name,'Sheet1')
    writer.save() 

##      
if __name__=='__main__':
    # 提取
    url_profession='https://www.jobmd.cn/work/Medical_imaging_radiology.htm?pge='
    url=[gen_Url(i+1,url_profession) for i in range(25)]
    Info,http=extract_allPage(url)
    # 筛选...
    Http,info=screen_info(http,Info)
    # fetch detail...
    df_detail=fetch_detail_all(Http)#每个单位的信息
    # 组合信息到pandas格式   
    df=concat_allInfo(Http,info,df_detail)
    df.index=np.arange(0,len(df),1)
    # save to excel
    dropIndex=[]
    for i in range(len(df)):
        try:
            dd=df.iloc[i,:]
            dd.to_excel('justForChecking'+str(i)+'.xlsx') 
        except:
            dropIndex.append(i)
            print('第{}条招聘信息有非法字符串'.format(i))        
    df=df.drop(dropIndex)
    df.to_excel('allJobs.xlsx') 
    print('完成！')