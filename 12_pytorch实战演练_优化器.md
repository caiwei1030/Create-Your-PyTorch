# 优化器

---

**优化器（Optimizer）** 是深度学习中用于**更新模型参数以最小化损失函数**的核心组件。

它通过反向传播计算得到的梯度，调整参数值，逐步优化模型性能。

---

### 一、核心作用
1. **参数更新**  
   根据损失函数的梯度调整模型参数，使预测结果更接近真实值。
2. **收敛加速**  
   通过动量（Momentum）、自适应学习率等策略加快训练收敛速度。
3. **避免局部最优**  
   引入随机性（如SGD的小批量更新）或自适应机制（如Adam），跳出局部极小值。

---

### 二、常见优化器类型
| 优化器                | 特点                         | 适用场景            |
| --------------------- | ---------------------------- | ------------------- |
| **SGD**               | 基础优化器，需手动调学习率   | 简单模型、调参实验  |
| **SGD with Momentum** | 引入动量加速收敛，缓解震荡   | 深层网络、非凸优化  |
| **Adam**              | 自适应学习率，默认效果较好   | 大多数深度学习任务  |
| **RMSprop**           | 解决Adagrad学习率骤降问题    | RNN、非平稳目标函数 |
| **Adagrad**           | 自动调整学习率，适合稀疏数据 | 自然语言处理        |

---

### 三、PyTorch中的基本用法
#### 1. 初始化优化器
```python
import torch.optim as optim

# 定义模型
model = nn.Linear(10, 2)

# 选择优化器（以Adam为例）
optimizer = optim.Adam(
    model.parameters(),   # 传入需优化的参数
    lr=0.001,             # 学习率
    betas=(0.9, 0.999),   # Adam的动量参数
    weight_decay=0.01     # L2正则化系数
)
```

#### 2. 单步训练流程
```python
for inputs, labels in dataloader:
    # 清零梯度（防止累积）
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
```

---

### 四、参数详解（以Adam为例）
| 参数           | 描述            | 默认值       | 调参建议                       |
| -------------- | --------------- | ------------ | ------------------------------ |
| `lr`           | 初始学习率      | 0.001        | 常用1e-3到1e-5，需根据任务调整 |
| `betas`        | 动量衰减系数    | (0.9, 0.999) | 通常无需修改                   |
| `eps`          | 数值稳定项      | 1e-8         | 避免除零，一般固定             |
| `weight_decay` | L2正则化系数    | 0            | 防止过拟合，常用0.01~0.1       |
| `amsgrad`      | 使用AMSGrad变体 | False        | 解决Adam收敛问题               |

---

### 五、高级技巧
#### 1. 分层学习率设置
```python
# 为不同层设置不同学习率
optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},  # 特征提取层
    {'params': model.classifier.parameters(), 'lr': 0.01}   # 分类层
], momentum=0.9)
```

#### 2. 学习率调度（Learning Rate Scheduling）
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 每个epoch后更新学习率
```

#### 3. 梯度裁剪（防梯度爆炸）
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 六、优化器选择策略
| 场景             | 推荐优化器     | 理由                                 |
| ---------------- | -------------- | ------------------------------------ |
| **新手入门**     | Adam           | 超参数鲁棒，默认效果好               |
| **计算机视觉**   | SGD + Momentum | 配合适当学习率调度，可能获得更优结果 |
| **自然语言处理** | AdamW          | 改进的Adam，更好处理权重衰减         |
| **强化学习**     | RMSprop        | 适合非平稳目标函数                   |
| **小批量数据**   | Adagrad        | 自适应稀疏特征                       |

---

### 七、常见问题及解决
#### 1. **训练震荡不收敛**
- **可能原因**：学习率过大、未使用动量。
- **解决**：  
  - 降低学习率  
  - 使用Adam或SGD with Momentum  
  - 增加`weight_decay`参数

#### 2. **收敛速度慢**
- **可能原因**：学习率过小、未启用自适应机制。
- **解决**：  
  - 增大学习率  
  - 换用Adam/RMSprop  
  - 添加学习率预热（Warmup）

#### 3. **过拟合**
- **解决**：  
  - 增大`weight_decay`（L2正则化）  
  - 使用早停法（Early Stopping）  
  - 添加Dropout层

---

### 八、代码示例：完整训练循环
```python
import  torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import  tqdm

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(test_data,batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        output=self.model1(x)
        return output

writer=SummaryWriter("Logs")
model=MyModel()
loss=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in tqdm(range(20),desc="epoch"):
    sum_loss=0
    for data in data_loader:
        imgs,targets=data
        output=model(imgs)
        res_loss=loss(output,targets)

        optim.zero_grad()
        res_loss.backward()
        optim.step()

        sum_loss=sum_loss+res_loss
    # 记录整个epoch的平均损失
    avg_loss = sum_loss / len(data_loader)
    writer.add_scalar("loss", avg_loss, epoch)
writer.close()
```

---

### 九、总结
- **核心作用**：通过梯度下降算法迭代更新模型参数，最小化损失函数。
- **关键选择**：  
  - 默认首选Adam，进阶任务尝试SGD+Momentum  
  - 根据任务特性调整学习率和正则化参数  
- **最佳实践**：  
  - 始终在训练前清零梯度（`zero_grad()`）  
  - 配合学习率调度器提升性能  
  - 监控梯度范数防止爆炸/消失  

理解不同优化器的特性并合理使用，能够显著提升模型训练效率和最终性能。