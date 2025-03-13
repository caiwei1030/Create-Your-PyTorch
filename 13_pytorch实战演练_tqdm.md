# Tqdm

>  tqdm是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息,用户只需要封装任意的迭代器，是一个快速、扩展性强的进度条工具库。

## 1.安装

```
pip install tqdm
1.传入可迭代对象
```

## 2.使用方法

### 1.经典使用方法

```
import time
from tqdm import *
for i in tqdm(range(1000)):
    time.sleep(.01)   #进度条每0.01s前进一次，总时间为1000*0.01=10s 
```

**运行结果如下**

100%|██████████| 1000/1000 [00:10<00:00, 93.21it/s]  
**使用trange**

```
trange(i) 是 tqdm(range(i)) 的简单写法

from tqdm import trange

for i in trange(1000):
    time.sleep(.01)
```

**运行结果如下**

100%|██████████| 1000/1000 [00:10<00:00, 93.21it/s]  

### 2.**为进度条设置描述**

在for循环外部初始化tqdm，可以打印其他信息：

```
import time
from tqdm import tqdm

pbar = tqdm(["a","b","c","d"])

for char in pbar:
    pbar.set_description("Processing %s" % char) # 设置描述
    time.sleep(1)  # 每个任务分配1s
```

​    **结果如下**

  0%|          | 0/4 [00:00<?, ?it/s]

Processing a:   0%|          | 0/4 [00:00<?, ?it/s]

Processing a:  25%|██▌       | 1/4 [00:01<00:03,  1.01s/it]

Processing b:  25%|██▌       | 1/4 [00:01<00:03,  1.01s/it]

Processing b:  50%|█████     | 2/4 [00:02<00:02,  1.01s/it]

Processing c:  50%|█████     | 2/4 [00:02<00:02,  1.01s/it]

Processing c:  75%|███████▌  | 3/4 [00:03<00:01,  1.01s/it]

Processing d:  75%|███████▌  | 3/4 [00:03<00:01,  1.01s/it]

Processing d: 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]

### 3.**手动控制进度**

```
import time
from tqdm import tqdm

with tqdm(total=200) as pbar:
    for i in range(20):
        pbar.update(10)
        time.sleep(.1)
```

**结果如下，一共更新了20次**

0%|          | 0/200 [00:00<?, ?it/s]

 10%|█         | 20/200 [00:00<00:00, 199.48it/s]

 15%|█▌        | 30/200 [00:00<00:01, 150.95it/s]

 20%|██        | 40/200 [00:00<00:01, 128.76it/s]

 25%|██▌       | 50/200 [00:00<00:01, 115.72it/s]

 30%|███       | 60/200 [00:00<00:01, 108.84it/s]

 35%|███▌      | 70/200 [00:00<00:01, 104.22it/s]

 40%|████      | 80/200 [00:00<00:01, 101.42it/s]

 45%|████▌     | 90/200 [00:00<00:01, 98.83it/s] 

 50%|█████     | 100/200 [00:00<00:01, 97.75it/s]

 55%|█████▌    | 110/200 [00:01<00:00, 97.00it/s]

 60%|██████    | 120/200 [00:01<00:00, 96.48it/s]

 65%|██████▌   | 130/200 [00:01<00:00, 96.05it/s]

 70%|███████   | 140/200 [00:01<00:00, 95.25it/s]

 75%|███████▌  | 150/200 [00:01<00:00, 94.94it/s]

 80%|████████  | 160/200 [00:01<00:00, 95.08it/s]

 85%|████████▌ | 170/200 [00:01<00:00, 93.52it/s]

 90%|█████████ | 180/200 [00:01<00:00, 94.28it/s]

 95%|█████████▌| 190/200 [00:01<00:00, 94.43it/s]

100%|██████████| 200/200 [00:02<00:00, 94.75it/s]

### **4.tqdm的write方法**

```
bar = trange(10)
for i in bar:
    time.sleep(0.1)
    if not (i % 3):
        tqdm.write("Done task %i" % i)
```

### **5.手动设置处理的进度**

通过update方法可以控制每次进度条更新的进度：

```
from tqdm import tqdm 
import time
#total参数设置进度条的总长度
with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.1)
        pbar.update(1)  #每次更新进度条的长度
```

#结果
  0%|          | 0/100 [00:00<?, ?it/s]
  1%|          | 1/100 [00:00<00:09,  9.98it/s]
  2%|▏         | 2/100 [00:00<00:09,  9.83it/s]
  3%|▎         | 3/100 [00:00<00:10,  9.65it/s]
  4%|▍         | 4/100 [00:00<00:10,  9.53it/s]
  5%|▌         | 5/100 [00:00<00:09,  9.55it/s]
  ...
  100%|██████████| 100/100 [00:10<00:00,  9.45it/s]
除了使用with之外，还可以使用另外一种方法实现上面的效果：

```
from tqdm import tqdm
import time

#total参数设置进度条的总长度
pbar = tqdm(total=100)
for i in range(100):
  time.sleep(0.05)
  #每次更新进度条的长度
  pbar.update(1)
#别忘了关闭占用的资源
pbar.close()
```

### **6.自定义进度条显示信息**

通过set_description和set_postfix方法设置进度条显示信息：

```
from tqdm import trange
from random import random,randint
import time

with trange(10) as t:
  for i in t:
    #设置进度条左边显示的信息
    t.set_description("GEN %i"%i)
    #设置进度条右边显示的信息
    t.set_postfix(loss=random(),gen=randint(1,999),str="h",lst=[1,2])
    time.sleep(0.1)
```




