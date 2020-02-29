# -*- coding: utf-8 -*-
import jieba
import wordcloud
txt = "董梦实是个小屁孩小猪土鸡蛋"
txt = jieba.lcut(txt)
txt = " ".join(txt)
w = wordcloud.WordCloud( \
    width = 1000, height = 700,\
    background_color = "white",
    font_path = "msyh.ttc"    
        )
w.generate(txt)
w.to_file("grwordcloud.png")

