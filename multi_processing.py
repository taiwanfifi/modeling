#!/usr/bin/env python
# coding: utf-8


import multiprocessing
import time
   
from datetime import datetime


def square(x):
    time.sleep(2)
    return x * x
   
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)
    print(datetime.now().time())
    inputs = [0,1,2,3,4]
    outputs = pool.map(square, inputs)
    print(datetime.now().time())
    print("Input: {}".format(inputs))
    print("Output: {}".format(outputs))
    print(datetime.now().time())



import multiprocessing
import time
  
  
def square(x):
    return x * x
  
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    inputs = [0,1,2,3,4]
    outputs_async = pool.map_async(square, inputs)
    outputs = outputs_async.get()
    print("Output: {}".format(outputs))