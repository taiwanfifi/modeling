"""from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

"""

from multiprocessing import Process, Pool
import os, time


def main_map(i):
    result = i * i
    time.sleep(1)
    return result


if __name__ == '__main__':
    inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      # 設定處理程序數量
    pool = Pool(4)

      # 運行多處理程序
    pool_outputs = pool.map(main_map, inputs)

      # 輸出執行結果
    print(pool_outputs)



"""import multiprocessing
import time
  
def square(x):
    return x * x
  
pool = multiprocessing.Pool()
inputs = [0,1,2,3,4]
outputs_async = pool.map_async(square, inputs)
outputs = outputs_async.get()
print("Output: {}".format(outputs))"""