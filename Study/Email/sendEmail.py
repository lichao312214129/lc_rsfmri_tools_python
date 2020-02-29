# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:57:21 2018

@author: lenovo
"""
from smtplib import SMTP_SSL
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import random
import threading
# ======================================================
mail_info = {
    "from": "lichao19870617@163.com",
    "to": "lichao19870617@163.com",
    "hostname": "smtp.163.com",
    "username":"lichao19870617@163.com",
    "password": "",
    "mail_subject": "test",
    "mail_text":"这只是一个测试",
    "mail_encoding": "utf-8"
}

def main():
    smtp = SMTP_SSL(mail_info["hostname"])
    smtp.set_debuglevel(1)
    smtp.ehlo(mail_info["hostname"])
    smtp.login(mail_info["username"], mail_info["password"])
    msg = MIMEText(mail_info["mail_text"], "plain", mail_info["mail_encoding"])
    msg["Subject"] = Header(mail_info["mail_subject"], mail_info["mail_encoding"])
    msg["from"] = mail_info["from"]
    msg["to"] = mail_info["to"]
    
    smtp.sendmail(mail_info["from"], mail_info["to"], msg.as_string())

    smtp.quit()




def start():
    try:
        main() 
    except Exception as e:
        print (e)
        start()

# ========================================================
# 单个发送
if __name__=='__main__':
    start()

# 多线程发送
#threads = []
#
#start()
#for i in range(5):
#    t = threading.Thread(target=start)
#    t.start()
#    threads.append(t)
#for t in threads:
#    t.join()