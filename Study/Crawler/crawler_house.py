# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 09:59:22 2018
安居客,沈阳新房
@author: lenovo
"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time
import pandas as pd
#
# 进入网站
browser=webdriver.Firefox()
browser.get("https://shen.fang.anjuke.com/?from=navigation")
# 将关键词输入搜索框，并点击搜索
input=browser.find_element_by_class_name("header-search-input")
input.clear()
input.send_keys("虾")

#newUrl=browser.find_element_by_xpath("//div[@class='header-search-hasinput']/ul/li")
search_button=browser.find_element_by_class_name("header-search-btn")#点击search
search_button.click()

#获得商家名

current_window =browser.current_window_handle #获取当前页面
browser.switch_to.window(current_window)# 更新到搜索后的页面
current_hanle=browser.current_window_handle#当前页面句柄名
#注意element 和elements是不一样的

allName=browser.find_elements_by_xpath("//div[@class='common-list-main']/*")
allname=[allName_.text for allName_ in allName]
a=allname[0]
allname=[a.split('\n') for a in allname]
allname=['|'.join(a) for a in allname]
allname=pd.DataFrame(allname)
allname.to_excel('aa.xlsx')
# url
links=browser.find_elements_by_xpath("//div[@class='default-list-item clearfix']/a[@href]")
allUrl=[link.get_attribute('href') for link in links]
