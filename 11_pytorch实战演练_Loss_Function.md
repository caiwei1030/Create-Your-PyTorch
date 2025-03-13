# Loss_Function

```
import  torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten,Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
epoch=0
for data in data_loader:
    imgs,targets=data
    output=model(imgs)
    res_loss=loss(output,targets)
    res_loss.backward()
    writer.add_scalar("loss",res_loss,epoch)
    epoch+=1
writer.close()
```



---

**损失函数（Loss Function）** 是机器学习和深度学习的核心组件，用于**量化模型预测结果与真实标签的差异**，为优化算法提供调整参数的依据。

---

### 一、核心作用
1. **性能评估**  
   数值化衡量模型预测的准确程度（值越小表示预测越接近真实值）。

2. **指导参数优化**  
   通过反向传播计算梯度，引导优化器（如SGD、Adam）调整模型参数。

3. **任务适配**  
   不同任务需选择不同的损失函数（如分类用交叉熵，回归用均方误差）。

---

### 二、常见损失函数分类
#### 1. 回归任务
| 损失函数                | 公式                                                         | 特点         | PyTorch实现类 |
| ----------------------- | ------------------------------------------------------------ | ------------ | ------------- |
| **均方误差（MSE）**     | \( $\frac{1}{N}\sum (y_{\text{pred}} - y_{\text{true}})^2$ \) | 对离群值敏感 | `nn.MSELoss`  |
| **平均绝对误差（MAE）** | \( $\frac{1}{N}\sum \|y_{\text{pred}} - y_{\text{true}}\|$ \) | 抗离群值     | `nn.L1Loss`   |

#### 2. 分类任务
| 损失函数                        | 适用场景                  | PyTorch实现类                         |
| ------------------------------- | ------------------------- | ------------------------------------- |
| **交叉熵损失（Cross-Entropy）** | 多分类（互斥类别）        | `nn.CrossEntropyLoss`                 |
| **二元交叉熵（BCE）**           | 二分类或多标签分类        | `nn.BCELoss` / `nn.BCEWithLogitsLoss` |
| **KL散度**                      | 概率分布差异度量（如GAN） | `nn.KLDivLoss`                        |

#### 3. 特殊任务
| 损失函数                    | 应用场景                 | PyTorch实现类          |
| --------------------------- | ------------------------ | ---------------------- |
| **对比损失（Contrastive）** | 相似性学习（如人脸识别） | 需自定义               |
| **Triplet损失**             | 特征嵌入（如推荐系统）   | `nn.TripletMarginLoss` |
| **Focal Loss**              | 类别不平衡（如目标检测） | 需自定义               |

---

### 三、PyTorch中的用法
#### 1. 基本流程
```python
import torch.nn as nn

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 前向计算
outputs = model(inputs)          # 模型预测输出
loss = criterion(outputs, labels)  # 计算损失值

# 反向传播优化
optimizer.zero_grad()
loss.backward()                   # 计算梯度
optimizer.step()                  # 更新参数
```

#### 2. 关键参数详解（以CrossEntropyLoss为例）
| 参数           | 作用                                  | 示例值                     |
| -------------- | ------------------------------------- | -------------------------- |
| `weight`       | 类别权重（处理不平衡数据）            | `torch.tensor([0.1, 0.9])` |
| `ignore_index` | 忽略指定类别                          | `-100`                     |
| `reduction`    | 聚合方式（`'mean'`/`'sum'`/`'none'`） | `'mean'`                   |

**示例代码**：
```python
# 处理类别不平衡
class_weights = torch.tensor([0.2, 0.8])  # 假设类别0样本少
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

### 四、输入输出规范
#### 1. 输入形状要求
| 损失函数            | 预测值形状 | 真实值形状 | 说明                             |
| ------------------- | ---------- | ---------- | -------------------------------- |
| `CrossEntropyLoss`  | `(N, C)`   | `(N,)`     | C为类别数，真实值为类别索引      |
| `BCEWithLogitsLoss` | `(N, *)`   | `(N, *)`   | 预测值未经过Sigmoid，真实值为0/1 |
| `MSELoss`           | `(N, *)`   | `(N, *)`   | 任意相同形状                     |

#### 2. 输出计算
- **单值标量**：默认`reduction='mean'`，返回批次平均损失。  
- **完整张量**：`reduction='none'`时返回每个样本的损失值。

---

### 五、实际应用技巧
#### 1. 损失函数选择策略
| 任务类型                 | 推荐损失函数        | 注意事项          |
| ------------------------ | ------------------- | ----------------- |
| **多分类**               | `CrossEntropyLoss`  | 输出层无需Softmax |
| **多标签分类**           | `BCEWithLogitsLoss` | 每个标签独立判断  |
| **回归（高斯噪声）**     | `MSELoss`           | 对异常值敏感      |
| **回归（拉普拉斯噪声）** | `L1Loss`            | 更鲁棒            |

#### 2. 自定义损失函数
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 避免log计算
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
```

#### 3. 多任务损失加权
```python
loss1 = criterion1(output1, target1) * weight1
loss2 = criterion2(output2, target2) * weight2
total_loss = loss1 + loss2
```

---

### 六、常见错误及解决
#### 1. 形状不匹配
- **错误信息**：`RuntimeError: size mismatch`  
- **示例**：使用CrossEntropyLoss时真实标签形状为`(N,1)`而非`(N,)`  
- **修正**：  
  ```python
  labels = labels.squeeze()  # 从 (N,1) 变为 (N,)
  ```

#### 2. 未正确处理概率
- **错误代码**：  
  ```python
  outputs = torch.softmax(model(inputs), dim=1)  # 错误：CrossEntropyLoss内部已含Softmax
  criterion = nn.CrossEntropyLoss()
  loss = criterion(outputs, labels)
  ```
- **修正**：直接传入Logits（无需手动Softmax）。

#### 3. 梯度爆炸
- **现象**：损失值变为NaN。  
- **解决**：  
  - 梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`  
  - 调整学习率  

---

### 七、可视化与监控
#### 1. TensorBoard 监控
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for epoch in range(epochs):
    loss = train_one_epoch(...)
    writer.add_scalar('Loss/train', loss, epoch)
```

#### 2. 损失曲线分析
- **健康训练**：损失平滑下降，最终收敛。  
- **异常情况**：  
  - 剧烈震荡 → 学习率过大  
  - 长期不下降 → 模型容量不足或学习率过小  

---

### 八、总结
**核心作用**：  
- 指导模型优化方向  
- 量化模型性能  

**选择原则**：  
- 匹配任务类型  
- 考虑数据特性（如类别平衡性、噪声分布）  

**PyTorch最佳实践**：  
- 优先使用内置损失函数  
- 自定义损失时继承`nn.Module`  
- 监控损失曲线调整超参数  

理解损失函数的设计逻辑，能帮助你针对具体问题选择或设计合适的评估标准，从而提升模型性能。