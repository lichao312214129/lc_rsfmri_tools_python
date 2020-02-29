# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:33:12 2018

@author: lenovo
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time
#======================================
#browser=webdriver.Chrome()
browser=webdriver.Firefox()
#browser=webdriver.Safari()
#browser=webdriver.Edge()
#browser=webdriver.PhantomJS()
#browser.get("http://www.taobao.com")
#print(browser.page_source)
#======================================
# 单个元素
browser=webdriver.Firefox()
browser.get("http://www.taobao.com")
input_first=browser.find_element_by_id("q")
input_second=browser.find_element_by_css_selector("#q")
input_third=browser.find_element(By.ID,"q")
print(input_first,input_second,input_first)
browser.close()
#======================================
# 多个元素
browser=webdriver.Firefox()
browser.get("http://www.taobao.com")
lis=browser.find_element_by_css_selector("li")
lis_c=browser.find_element(By.CSS_SELECTOR,"li")
print(lis,lis_c)
browser.close()
#======================================
#交互:pubmed例子
browser=webdriver.Firefox()
browser.get("https://www.ncbi.nlm.nih.gov/pubmed/?term=schizophrenia+AND+(resting-state+fmri)")
input=browser.find_element_by_id("term")
input.send_keys("schizophrenia AND (resting-state fmri)")
browser.find_element_by_id("search").click()#点击search
# 发表时间范围
# start
browser.find_element_by_id("facet_date_rangeds1").click()
browser.find_element_by_id("facet_date_st_yeards1").send_keys("2018")
browser.find_element_by_id("facet_date_st_monthds1").send_keys("06")
browser.find_element_by_id("facet_date_st_dayds1").send_keys("01")
# end
browser.find_element_by_id("facet_date_end_yeards1").send_keys("2018")
browser.find_element_by_id("facet_date_end_monthds1").send_keys("09")
browser.find_element_by_id("facet_date_end_dayds1").send_keys("01")
# apply
browser.find_element_by_id("facet_date_range_applyds1").click()
#============================================================
time.sleep(10)
input.clear()
input.send_keys("iPad")
button=browser.find_element_by_class_name("btn-search")
button.click()
time.sleep(10)
browser.close()
#========================================================
browser=webdriver.Firefox()
browser.get("https://www.baidu.com/")
browser.find_element_by_xpath("//*[@id='kw']").send_keys("selenium")
browser.close()
#======================================
#12306购票
# login
browser=webdriver.Firefox()
browser.get("https://kyfw.12306.cn/otn/login/init")
img_match=browser.find_elements_by_xpath("//div[@class='touclick-hov touclick-bgimg']\
                                         [@style='left: 52px; top: 133px;']")
# 点购票
li=browser.find_elements_by_xpath("//div[@id='indexLeftBL']/ul[@class='leftItem']/li")
li[2].click()
# 输入城市和时间
 #1 登陆
## close
browser.close()
#======================================
#把动作附加到交互链中
from selenium.webdriver import ActionChains
browser=webdriver.Firefox()
url="https://fanyi.baidu.com/?aldtype=16047#en/zh/concerned"
browser.get(url)
#切换到目标元素所在的frame
browser.switch_to.frame("iframeResult")
#确定拖拽目标的起点
source=browser.find_element_by_id("draggable")
#确定拖拽目标的终点
target=browser.find_element_by_id("droppable")
#形成动作链
actions=ActionChains(browser)
actions.drag_and_drop(source,target)
#执行
actions.perform()
'''
1.先用switch_to_alert()方法切换到alert弹出框上
2.可以用text方法获取弹出的文本 信息
3.accept()点击确认按钮
4.dismiss()相当于点右上角x，取消弹出框
'''
t=browser.switch_to_alert()
print(t.text)
t.accept()
time.sleep(10)
browser.close()
browser.close()
#======================================
#7：执行javascript
#下面的例子是执行就是，拖拽进度条到底，并弹出提示框
from selenium import webdriver
browser=webdriver.Firefox()
browser.get("https://www.zhihu.com/explore")
browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
browser.execute_script("alert('To Button')")
browser.close()
#======================================
# 获取属性
browser=webdriver.Firefox()
url="https://www.zhihu.com/explore"
browser.get(url)
logo=browser.find_element_by_id("zh-top-link-logo")
print(logo)
print(logo.get_attribute("class"))
browser.close()
#======================================
# 获取文本值
browser=webdriver.Firefox()
url="https://www.zhihu.com/explore"
browser.get(url)
logo=browser.find_element_by_id("zh-top-link-logo")
print(logo)
print(logo.text)
browser.close()
#======================================
#获取ID、位置、大小和标签名
browser=webdriver.Firefox()
url="https://www.zhihu.com/explore"
browser.get(url)
logo=browser.find_element_by_id("zh-top-link-logo")
print(logo)
#id
print(logo.id)
#位置
print(logo.location)
#标签名
print(logo.tag_name)
#大小
print(logo.size)
browser.close()
#======================================
#隐式等待
browser=webdriver.Firefox()
browser=webdriver.Chrome()
url="https://www.zhihu.com/explore"
browser.get(url)
browser.implicitly_wait(10)
logo=browser.find_element_by_id("zh-top-link-logo")
print(logo)
browser.close()
#======================================



try:
    browser.get("https://www.baidu.com")
    input=browser.find_element_by_id("kw")
    input.send_keys("Python")
    input.send_keys(Keys.ENTER)
    wait=WebDriverWait(browser,10)
    wait.until(EC.presence_of_element_located((By.ID,"content_left")))
    print(browser.current_url)
    print(browser.get_cookies())
    print(browser.page_source)
    time.sleep(10)
finally:
    browser.close()