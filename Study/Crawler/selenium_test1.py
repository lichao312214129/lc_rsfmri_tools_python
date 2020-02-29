# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:50:53 2018

@author: lenovo
"""

from selenium import webdriver
import time
 
drivers = ['HtmlUnit', 'PhantomJS', 'Chrome', 'FF', 'IE'] 
 
dervers_time = {
	'HtmlUnit' : 0,
	'PhantomJS' : 0,
	'Chrome' : 0,
	'FF' : 0,
	'IE' : 0,
}
times = 50
def run_with_Chrome():
	common_step(webdriver.Chrome())
 
def run_with_FF():
	common_step(webdriver.Firefox())
	
def run_with_IE():
	common_step(webdriver.Ie())
 
def run_with_PhantomJS():
	common_step(webdriver.PhantomJS(executable_path=r'C:\Python27\Scripts\phantomjs.exe'))
	
def run_with_HtmlUnit():
	driver = webdriver.Remote("http://localhost:4444/wd/hub", 
								desired_capabilities=webdriver.DesiredCapabilities.HTMLUNIT)
	common_step(driver)
	
def common_step(driver):
	driver.get('http://www.baidu.com')
	ele = driver.find_element_by_id('su')
	print (ele.get_attribute('value'))
	driver.quit()
 
for i in range(times):
	print ('=============Times %s============' % i)
	for driver in drivers:
		start = time.time() 
		print (start)
		eval('run_with_%s()'%driver)
		end = time.time() 
		print (end)
		elapse_time = end-start
		dervers_time[driver] += elapse_time
		print ('elapse for %s:%s' % (driver, elapse_time))
	
for k,v in dervers_time.items():
	print ('avg elapse for %s in %s times:%s' % (k, times, v/times))