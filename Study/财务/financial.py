# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:14:22 2018
财务
@author: Li Chao
"""
# import
#import nipype
#import sys
#sys.path.append(r'D:\myCodes\MVPA_LC\Python\MVPA_Python\utils')
from lc_selectFile import selectFile
from lc_saveDataFrameToExcel import saveOneToExcel
from obtainOneItemContent import obtainOneItemContent
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
import os

##
#touZiRen=r'D:\其他\老舅财务\名单.xlsx'
#targetItem=['户名',	'帐号',	'交易日期',	'摘要',\
#  '借贷标志','交易金额']
#otherTargetItem=[-4,-3]# 投资人和收返利卡号有的显示错误，因此用索引来寻找
    
## def
def main(rootPath,n_jobs):
    start=time.time()

    #============== 等号之间的为需要自定义的参数==================
    # n_jobs为线程数，根据自己的电脑选择适合的数目
#    rootPath=r'D:\其他\老舅财务\allData'
    batchSize=100 # 对于每个线程，每次处理的人数（此处为初始值，后面根据人数会自动调整）
    resultFolder=r'D:\其他\老舅财务'#保存结果路径
    touZiRen=r'D:\其他\老舅财务\名单.xlsx' # 投资人名单
    """
    注意：
    以下两个item是需要提取的项目名称以及在表头的位置，因为有的txt的项目显示为？，
    所以添加一个为otherTargetItem的位置列表，用来提取没有名字的项目
    """
    targetItem=['户名',	'帐号',	\
                '交易日期',	'摘要',\
                '借贷标志','交易金额'] #
    otherTargetItem=[-4,-3]
    # 保存到excel后的项目名称
    itemName_toSave=['户名',	'帐号',	'交易日期',	'摘要',\
                       '借贷标志','交易金额','投资人','收返利卡号']
    #============== 等号之间的为需要自定义的参数==================
    
    
    ## 获取所有Text文档路径
    allTxtFile=selectFile(rootPath) 
    
    ## make save folder   
    try:
        saveFolder_path=os.path.join(resultFolder,'financial_Results')
        os.mkdir(saveFolder_path)
    except FileExistsError:
        print('already have result folder')
        
    ## 抽取所有数据
    allContent=extractData_allTxt(allTxtFile,touZiRen,targetItem,otherTargetItem,\
                                  itemName_toSave)
    
    ## 整理所有数据
    # 整理投资人项目的内容（名单，有的投资人不在txt中，此时空列表将会被删除）
    unique_allName_1d,allName_noSpace=\
      obtainOneItemContent(allContent,itemName='投资人')
    # 整理所有的数据，耗时比较长
    uniNameLen=len(unique_allName_1d)
    # 根据人数再次设置batchSize
    if uniNameLen<1000:
        batchSize=len(unique_allName_1d)
    #确定blocks 数目
    if uniNameLen != batchSize:
        N_blocks=max(np.arange(0,uniNameLen,batchSize))
        myRange=np.arange(0,uniNameLen,batchSize)
    else:
        N_blocks=uniNameLen
        myRange=[0]
    
    sorted_content=Parallel(n_jobs,backend='threading')\
        (delayed(concatDataFrame)(allContent,unique_allName_1d,allName_noSpace,\
         batchSize,N_blocks,block)\
         for block in myRange)
     #    sorted_Content,Name=concatDataFrame(allContent)
    
    ## 保存所有数据为excel格式（先cd到保存结果的文件夹）
    s=time.time()
    print('整理完毕，开始保存到excel中......')
    i_s=0
    i_e=len(sorted_content[0])
    for c in sorted_content:
        print('正在保存第{}/{}批投资人'.format(i_s,N_blocks))
        name=unique_allName_1d[i_s:i_e]
        saveAllToExcel(c,name,saveFolder_path)
        i_s+=len(c)
        i_e+=len(c)        
    e=time.time()
    
    print('保存为excel所花费时间为{:.0f}秒\n\n'.format(e-s))
    print('======总运行时间为{:.0f}秒======\n\
          \r======结果保存在:[{}]======'.format\
          (np.ceil(e-start),saveFolder_path))
    


##
def saveAllToExcel(Content,name,saveFolder_path):
    i=0
    for content in Content:
        saveOneToExcel(content,os.path.join(saveFolder_path,name[i]+'.xlsx'))
        i+=1

#
def concatDataFrame(allContent,unique_allName_1d,allName_noSpace,
                    batchSize,N_blocks,block):
#    s=time.time()
    print('正在整理第{}/{} 批的投资人\r'.format(block,N_blocks))
#    allName=[]
#    for content in allContent:
#        allName.append([a['投资人'].iloc[0] for a in content])
#    # 把allName的空格去掉
#    allName_noSpace=[]
#    for Name in allName:
#        allName_noSpace.append([name.replace(' ','') for name in Name])
#    # 把all name拉成一列，并去掉空格，最后求unique name
#    allName_1d=reduce(operator.add, allName)
#    allName_1d=[name.replace(' ','') for name in allName_1d]
#    unique_allName_1d=np.unique(allName_1d)
    # loop concat
    # 求每个名字在allContent的loc，为后续的concat做准备
    n_row=len(allContent)
    Content=[]
    N_name=len(unique_allName_1d)
    s_point=block
    e_point=min(block+batchSize,N_name)
    count=0
    for name in unique_allName_1d[s_point:e_point]:
        if batchSize>=30:
            print('\t\t正在整理第{}/{}批投资人的第{}个人'.format(block,N_blocks,count))
        loc=[]
        for all_name in allName_noSpace:
#            if name in all_name:
            loc.append(myfind(name,all_name))
            concat_list=[]
            for row in range(n_row):
                try:
                   concat_list.append(allContent[row][loc[row][0]])
                except IndexError:
                   continue
            try:
                content=pd.concat(concat_list)
            except ValueError:
                continue
        count+=1
        Content.append(content)
#    e=time.time()
#    print('整合数据花费时间为 {}'.format(e-s))
    return Content

##
def extractData_allTxt(allTxtFile,\
                       touZiRen,\
                       targetItem,\
                       otherTargetItem,\
                       itemName_toSave):
    
    ##
    s=time.time()

    # loop
    Content=[]
    count=1
    N_txt=len(allTxtFile)
    for oneTxt in allTxtFile:
        print('正在抽取第 {}/{} 个text文档'.format(count,N_txt))
        content=extractData_oneTxt(oneTxt,touZiRen,\
                       targetItem,otherTargetItem,\
                       itemName_toSave)
        count+=1
        Content.append(content)
    e=time.time()
    print('抽取所有text文档时间为 {} '.format(e-s))
    return Content

##    
def extractData_oneTxt(textFile,touZiRen,\
                       targetItem,otherTargetItem,itemName_toSave):
    # read txt to pd
    df = pd.read_table(textFile, sep='[ \t]',engine='python',delimiter='|')
    # txt中的投资人name
    allName=df.iloc[:,-4]
    allName=allName.astype(str)
    allName=[name.replace(' ','') for name in allName]
    uni_allName =np.unique(allName)
    # item
    allItem=df.columns
    # 获得筛选的content
    loc_item=np.zeros([1,len(allItem)])
    for i in targetItem:
        loc_item=np.vstack([loc_item,allItem==i])
    loc_item=[bool(boo)  for boo in np.sum(loc_item,axis=0)]
    content=df.iloc[:,loc_item]
    content=[content,df.iloc[:,otherTargetItem]]
    content=pd.concat(content, axis=1)
    # 修改Content的列名
    content.columns = itemName_toSave
    
    # 求投资人和txt投资人的交集
    touZiRen=pd.read_excel(touZiRen)
    touZiRen=touZiRen.iloc[:,0]
#    touZiRen=touZiRen.astype(str)    
    matchName=[name for name in touZiRen[:] if name in uni_allName]

    # 定位投资人,一定要注意dtype要统一
    allName=pd.Series(allName)
    loc_touZiRen=[allName==l for l in matchName]
    # 抽取投资人的数据
    Content=[content.iloc[np.array(c),:] for c in loc_touZiRen]
#    e=time.time()
#    print('running time is {}'.format(e-s))
    return Content

##
def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

if __name__=='__main__':
    # 如果人数不多，n_jobs设置为1比较好。只有当人数非常多时，才设置为大于1的数
    main(rootPath=r'D:\其他\老舅财务\allData',n_jobs=1)