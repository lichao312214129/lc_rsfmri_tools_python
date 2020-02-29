# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 09:23:36 2018
查询excel表格中期刊的影响因子，并绘制柱状图
@author: Li Chao
"""
# from findSCIFactor import *
# -*- coding: utf-8 -*-
# import
import xlrd,os
import numpy as np
from matplotlib import pyplot as plt
# input
excel_path_2018='D:/myCodes/private_code/影响因子查询和对比/2017年影响因子_20180626.xlsx'
excel_path_2017='D:/myCodes/private_code/影响因子查询和对比/Journal+Impact+factor_2017.xlsx'
target_name='European journal of neuroscience'
def plot_bar():
    IF_2017, IF_2018=compare_IF()
    plt.figure(figsize=(5,5)) 
    fig=plt.bar([0,1],[IF_2017,IF_2018],fc='g')

    ## 设置显示参数
    # x 轴刻度
    plt.xticks([0,1],['2017','2018'],rotation=0)
    # title
    plt.title(target_name+"'s IF")
    # 刻度 size
    plt.tick_params(labelsize=15) 
    #设置横纵坐标的名称以及对应字体格式
#    font2 = {'family' : 'Times New Roman',
#    'weight' : 'normal',
#    'size'   : 15,
#    }
#    plt.xlabel(['2017','2018'],font2)
#    plt.ylabel('if',font2)
    
    # bar上加数字
    autolabel(fig)
#    plt.axis('off')
#    plt.savefig('if_compare.tif',format='tif', dpi=600)
    plt.show()

# bar上面显示数字。此函数下载至网络 
def autolabel(rects):
    # rects 为bar的句柄
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.01*height, '%s' % float(height))

def compare_IF():
    IF_2017=read_excel(excel_path_2017,target_name)
    IF_2018=read_excel(excel_path_2018,target_name)
    return IF_2017, IF_2018

def read_excel(excel_path,target_name):
    excel_path=os.path.abspath(excel_path)
    target_name=target_name.lower()
    ExcelFile=xlrd.open_workbook(excel_path)
    sheet=ExcelFile.sheet_by_index(0)  
    IF=sheet.col_values(4)[3:]#第二列内容
    j_name=sheet.col_values(1)[3:]#第二列内容
    j_name=[item.lower() for item in j_name]  
    if target_name in j_name:
        loc=j_name.index(target_name)
        IF_target=IF[loc]
        IF_target=float(IF_target)
        return IF_target
    else:
        print('您要查询的期刊:{}不在excel中'.format(target_name))        
if __name__ =='__main__':
#    from findSCIFactor import *
    IF_2017, IF_2018=compare_IF()
    with open("douban.txt","w") as f:
        f.write(str(IF_2018))
    f.close()
    plot_bar()