# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:36:18 2018

@author: lenovo
"""

import requests
import json
 
search_data = '今天天气不错'
headers = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'}
url = 'http://fanyi.baidu.com/basetrans'
data = {'query': search_data,
        'from': 'zh',
        'to': 'en'}
 
response = requests.post(url, headers=headers, data=data)
res = response.content.decode()
real_data = json.loads(res)['trans'][0]['dst']
print("{}的翻译结果是：{}".format(search_data, real_data))