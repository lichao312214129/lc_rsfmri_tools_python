# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:33:22 2018
微信自动回复信息，代码来自网络：https://blog.csdn.net/Cocktail_py/article/details/79765331
@author: lenovo
"""
# import
import itchat
import os
import re
import shutil
import time
from itchat.content import *
#from itchat.content import TEXT
# 代码实现如下：

@itchat.msg_register([TEXT])

def text_reply(msg):

    # 匹配任何文字和表情，然后自动回复

    match = re.search("(.*)",msg["Text"]).span()
    if match:
        itchat.send(("这是自动回复：不好意思，我现在没有看微信，看到后立刻回复您"),msg["FromUserName"])

# 匹配 图片，语言，视频，分享，然后自动回复
@itchat.msg_register([PICTURE,RECORDING,VIDEO,SHARING])
def other_reply(msg):
    itchat.send(("这是自动回复：不好意思，我现在没有看微信，看到后立刻回复您"),msg["FromUserName"])
#
if __name__ == "__main__":
    itchat.login()
    ########
    friends = itchat.get_friends(update=True)
#    print(friends)
    nick_name=list()
    for i in range(len(friends)):
        nick_name.append(friends[i].NickName)
    loc=nick_name.index('柔＆刚~徐克')
    tar_friend=friends[loc]
    ########
#    itchat.auto_login(enableCmdQR=True,hotReload=True)

    itchat.run()
    itchat.logout()