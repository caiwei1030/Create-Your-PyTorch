# Linear

```
import torch
import torchvision.datasets
from holoviews import output
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(test_data,batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

model=MyModel()

for data in data_loader:
    imgs,targets=data
    output=torch.flatten(imgs)
    output=model(output)
    print(output.shape)
```

---

`nn.Linear` 是 PyTorch 中实现全连接层（Fully Connected Layer）的核心组件，用于将输入特征的维度映射到目标输出维度。

它在神经网络中负责**全局特征整合**和**非线性变换前的线性投影**。

---

### 一、核心作用
1. **特征维度变换**  
   将输入向量的维度从 `in_features` 映射到 `out_features`，实现高维到低维（或低维到高维）的线性转换。

2. **全局信息整合**  
   通过权重矩阵的乘法操作，混合输入特征的所有维度信息。

3. **分类/回归输出**  
   通常作为网络的最后几层，输出分类概率或回归值（需配合激活函数如Softmax或Sigmoid）。

---

### 二、参数详解
| 参数           | 描述           | 示例值                     | 默认值 |
| -------------- | -------------- | -------------------------- | ------ |
| `in_features`  | 输入特征维度   | 784（如展平后的MNIST图像） | 必填   |
| `out_features` | 输出特征维度   | 10（分类类别数）           | 必填   |
| `bias`         | 是否添加偏置项 | `True`/`False`             | `True` |

---

### 三、输入输出形状
#### 1. 输入张量形状
- **格式**：`(*, in_features)`  
  - 支持任意额外维度（如批次维度、序列长度等）。  
  - **示例**：  
    - 单样本：`(in_features)`  
    - 批次数据：`(batch_size, in_features)`  
    - 序列数据：`(seq_len, batch_size, in_features)`

#### 2. 输出张量形状
- **格式**：`(*, out_features)`  
  - 保持输入的前导维度不变，仅最后一维变为 `out_features`。  
  - **示例**：  
    - 输入 `(32, 784)` → 输出 `(32, 256)`  
    - 输入 `(5, 10, 1024)` → 输出 `(5, 10, 512)`

---

### 四、代码示例
#### 1. 基本用法
```python
import torch.nn as nn

# 定义全连接层：输入维度100 → 输出维度50
fc = nn.Linear(in_features=100, out_features=50)

# 前向传播
input = torch.randn(32, 100)  # 批次大小32，输入维度100
output = fc(input)           # 输出形状 (32, 50)
```

#### 2. 构建多层感知机（MLP）
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),  # 输入784 → 输出256
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)     # 最终输出10类
        )
    
    def forward(self, x):
        # 展平图像数据：例如 (32,1,28,28) → (32,784)
        x = x.view(x.size(0), -1)
        return self.layers(x)
```

---

### 五、数学原理
**运算公式**：  
$$
\text{输出} = \text{输入} \times W^T + b
$$

- \(W\)：权重矩阵，形状为 `(out_features, in_features)`  
- \(b\)：偏置向量，形状为 `(out_features,)`（当 `bias=True` 时）  

**示例计算**：  
- 输入：`x = [x₁, x₂, x₃]`（`in_features=3`）  
- 权重：`W = [[w₁₁, w₁₂, w₁₃], [w₂₁, w₂₂, w₂₃]]`（`out_features=2`）  
- 输出：  
  $$
  y = [x₁w₁₁ + x₂w₁₂ + x₃w₁₃ + b₁,\quad x₁w₂₁ + x₂w₂₂ + x₃w₂₃ + b₂]
  $$

---

### 六、实际应用技巧
#### 1. 展平多维输入
在卷积网络后使用全连接层时，需将特征图展平为一维向量：
```python
# 输入形状 (batch_size, channels, height, width)
x = torch.randn(32, 64, 12, 12)
x = x.view(x.size(0), -1)  # 展平为 (32, 64*12*12) = (32, 9216)
fc = nn.Linear(9216, 256)
```

#### 2. 权重初始化
使用合适的初始化方法加速收敛（如He初始化）：
```python
nn.init.kaiming_normal_(fc.weight, mode='fan_in', nonlinearity='relu')
nn.init.constant_(fc.bias, 0.0)
```

#### 3. 配合正则化技术
- **Dropout**：防止过拟合  
  ```python
  nn.Sequential(
      nn.Linear(256, 128),
      nn.Dropout(p=0.5),  # 丢弃50%神经元
      nn.ReLU()
  )
  ```
- **BatchNorm**：稳定训练  
  ```python
  nn.Sequential(
      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU()
  )
  ```

---

### 七、常见错误及解决
#### 1. 输入维度不匹配
- **错误信息**：`RuntimeError: size mismatch`  
- **原因**：输入的最后一维不等于 `in_features`。  
- **检查步骤**：  
  ```python
  print(input.shape)  # 确认最后一维是否为in_features
  ```

#### 2. 未展平多维数据
- **错误示例**：  
  ```python
  # 输入形状 (32, 64, 12, 12) → 直接传入Linear(in_features=64)
  fc = nn.Linear(64, 10)
  output = fc(x)  # 报错：期望输入维度64，实际输入是64*12*12
  ```
- **修正**：先展平数据。

#### 3. 梯度消失/爆炸
- **现象**：训练损失不下降或变为NaN。  
- **解决**：  
  - 使用Batch Normalization  
  - 调整学习率  
  - 使用梯度裁剪（`torch.nn.utils.clip_grad_norm_`）

---

### 八、与卷积层的对比
| 特性         | `nn.Linear`                                             | `nn.Conv2d`                                                  |
| ------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| **连接方式** | 全连接（每个输入与输出相连）                            | 局部连接（滑动窗口）                                         |
| **参数数量** | \($ \text{in\_features} \times \text{out\_features} $\) | \( $\text{in\_channels} \times \text{out\_channels} \times \text{kernel\_size}^2 $\) |
| **适用场景** | 全局特征整合（如分类层）                                | 局部特征提取（如图像、序列）                                 |

---

### 九、总结
`nn.Linear` 的核心价值：
- **维度映射**：灵活调整特征维度，适配不同任务需求。
- **信息融合**：通过矩阵乘法整合全局特征。
- **模型输出**：作为分类/回归任务的最终决策层。

实际应用中需注意：
1. 输入数据需展平为一维向量（除非处理序列等特殊结构）。
2. 合理初始化权重和配合正则化技术以提高性能。
3. 监控训练过程，及时调整学习率或网络结构。