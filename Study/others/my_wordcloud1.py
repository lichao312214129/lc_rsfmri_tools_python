#GovRptWordCloudv2.py
import jieba
import wordcloud
from scipy.misc import imread
mask = imread("IU7A0290.JPG")
excludes = { }
txt = "董梦实是个小屁孩小猪土鸡蛋"
txt = jieba.lcut(txt)
txt = " ".join(txt)
w = wordcloud.WordCloud(\
    width = 1000, height = 700,\
    background_color = "white",
    font_path = "msyh.ttc", mask = mask
    )
w.generate(txt)
w.to_file("grwordcloudm.png")
