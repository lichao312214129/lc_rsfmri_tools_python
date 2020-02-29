# -*- coding: utf-8 -*-
import multiprocessing

def f(x):
    return x * x

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
xs = range(5)

# method 1: map
print (pool.map(f, xs))  # prints [0, 1, 4, 9, 16]

# method 2: imap
for y in pool.imap(f, xs):
    print (y)            # 0, 1, 4, 9, 16, respectively

# method 3: imap_unordered
for y in pool.imap_unordered(f, xs):
    print(y)           # may be in any order