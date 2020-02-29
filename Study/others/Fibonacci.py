# -*- coding: utf-8 -*-
def Fibonacci(N_mo):
    # 输入月份
#    N_mo=eval(input("请输入需要计算的月数：\t"))
#    N_mo=12
    dict = {}
    dict['0']=1
    dict['1'] =0
    dict['2'] = 0
    for mo in range(1,N_mo+1):
        dict['2']=dict['2']+dict['1']
        dict['1']=dict['0']
        dict['0']=dict['2']
    total=dict['0']+dict['1']+dict['2']
    return total
