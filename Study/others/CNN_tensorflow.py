# -*- coding: utf-8 -*-
import multiprocessing
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
import time

time_start=time.time() #time.time()为1970.1.1到当前时间的毫秒数 
for i in range(10000):
#    print('第{:d}次\r'.format(i));
    a=np.random.random(size=[100,1000])
time_end=time.time() #time.time()为1970.1.1到当前时间的毫秒数  
print('{:3f}'.format(time_end-time_start))