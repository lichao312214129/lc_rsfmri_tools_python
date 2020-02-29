# -*- coding: utf-8 -*-
#GovRptWordCloudv1.py
import jieba
import wordcloud
f = open("test.txt", "r")
 
t = f.read()
f.close()
ls = jieba.lcut(t)
txt=ls
txt = " ".join(ls)
w = wordcloud.WordCloud( \
    width = 1000, height = 700,\
    background_color = "white",
    font_path = "msyh.ttc",\
    max_words=20,max_font_size=200    
    )
w.generate(txt)
w.to_file("grwordcloud.png")