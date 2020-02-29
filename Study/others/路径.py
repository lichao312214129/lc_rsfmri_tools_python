import sys
import os
sys.platform #windows显示win32，linux显示linux
dir=os.listdir() #不给参数默认输出当前路径下所有文件
os.listdir('/home/python') #可以指定目录
######
path="D:\myMatlabCode\Python\AI-Practice-Tensorflow-Notes"
os.path.normcase("D:\myMatlabCode\Python\AI-Practice-Tensorflow-Notes") 
os.path.abspath(path) 
split=os.path.split(path) 
os.path.dirname(path) 
os.path.basename(path) 