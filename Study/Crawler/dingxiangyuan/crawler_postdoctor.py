d# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:17:47 2018

@author: lenovo
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time
import pandas as pd
import numpy as np
#
def open_startUrl(start_urls = 'https://www.jobmd.cn/work/Postdoctoral.htm'):
    browser=webdriver.Firefox()
    browser.get(start_urls )
    return browser

def parse(browser):
    allInfo = browser.find_element_by_xpath("//dl[@class='rm-dl mb30']")
    allInfo.get_attribute('href')
    
    name = allInfo.find_elements_by_xpath("//a[@class='rm-name']")
    name=[name_.text for name_ in name]
    
    Company = allInfo.find_elements_by_xpath("//a[@class='rm-company']")
    Company=[Company_.text for Company_ in Company]
    
    location = allInfo.find_elements_by_xpath("//li[@class='w-li2']/span")
    location=[location_.text for location_ in location]
    
    http=allInfo.find_elements_by_xpath("//li[@class='w-li4']/span/a[@href]")
    http=[http_.get_attribute('href') for http_ in http]
    
    # to DataFrame
    all_info=np.array([name,Company,location,http]).T
    all_info=pd.DataFrame(all_info,columns=['职位','单位','地点','链接'])
#    s=time.time()
#    all_info.to_excel(str(s)+'.xlsx')
    return browser,all_info


def nextPage(browser):
    # 先判断是存在最后一个是否为：上一页
    lastButton=browser.find_element_by_xpath("//ul[@class='pager']/li[7]").text
    if lastButton!='上一页':
        # next page
        nextPageButton=browser.find_element_by_class_name("pager-next")
        nextPageButton.click()
        # 更新页面
        current_window =browser.current_window_handle #获取当前页面
        browser.switch_to.window(current_window)# 更新到搜索后的页面
    else:
        print('已经是最后一页')
    return browser,lastButton

def main(lastButton=100):
    #open start url
    browser=open_startUrl(start_urls = 'https://www.jobmd.cn/work/Postdoctoral.htm')
    # crawler
    All_info=pd.DataFrame([],columns=['职位','单位','地点','链接'])
    while lastButton !='上一页':
        browser,all_info=parse(browser)
        All_info=pd.concat([All_info,all_info])
        time.sleep(5)
        browser,lastButton=nextPage(browser)
    All_info.to_excel('博士后招聘信息.xlsx')
    All_info.to_csv('博士后招聘信息.txt','|')
    browser.close()
    print('All done!')
    return All_info

if __name__=='__main__':
    All_info=main()