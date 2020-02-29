# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:02:09 2018

@author: lenovo
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time
#
import requests
from PIL import Image
from json import loads
import getpass
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# 禁用安全请求警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
#

# 获取验证码图片
def getImg():
    headers = {
            "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        }
    
    session = requests.session()
    url = "https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login&rand=sjrand";
    response = session.get(url=url,headers=headers,verify=False)
    # 把验证码图片保存到本地
    with open('img.jpg','wb') as f:
        f.write(response.content)
    # 用pillow模块打开并解析验证码,这里是假的，自动解析以后学会了再实现
    try:
        im = Image.open('img.jpg')
        # 展示验证码图片，会调用系统自带的图片浏览器打开图片，线程阻塞
        im.show()
        # 关闭，只是代码关闭，实际上图片浏览器没有关闭，但是终端已经可以进行交互了(结束阻塞)
        im.close()
    except:
        print (u'请输入验证码')
        
#===============================================================
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