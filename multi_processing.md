### ToC
- [處理程序數量](#models)



### process
建議使用處理程序 (process) 數量
可使用 multiprocessing.cpu_count() 或 os.cpu_count() 來獲取當前機器的 CPU 核心數量。
假設目前 CPU 是四核，那麼 process 設定如果超過 4 的話，代表有個核會同時運行 2 個以上的任務，而 CPU 之間程序處理會切換造成本進而降低處理效率，所以建議設置 process 時，最好等於當前機器的 CPU 核心數量。



```py
# method 1
import multiprocessing
cpus = multiprocessing.cpu_count()

# method 2
import os
cpus = os.cpu_count() 
```





