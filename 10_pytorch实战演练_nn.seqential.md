# nn.seqential

```
import  torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten,Linear
from torch.utils.tensorboard import SummaryWriter


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

model=MyModel()
input=torch.ones((64,3,32,32))
output=model(input)
print(output.shape)

writer=SummaryWriter("seq_logs")
writer.add_graph(model,input)
writer.close()
```

`nn.Sequential` 是 PyTorch 中用于**顺序组合多个网络层**的容器类，它能显著简化模型定义流程，尤其适用于线性堆叠的网络结构。

---

### 一、核心作用
1. **简化层堆叠**  
   将多个网络层按顺序封装为一个整体模块，避免手动编写逐层调用的代码。

2. **自动注册子模块**  
   内部的所有层会被自动注册为子模块（`submodule`），确保参数能被优化器识别和更新。

3. **直观结构展示**  
   通过`print(model)`可清晰查看网络层次结构，提升代码可读性。

---

### 二、基础用法
#### 1. 直接传入层实例
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),  # 第0层
    nn.ReLU(),             # 第1层
    nn.Linear(256, 10)     # 第2层
)

# 输出模型结构
print(model)
```
**输出**：
```
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```

#### 2. 使用 `add_module` 动态添加层
```python
model = nn.Sequential()
model.add_module("fc1", nn.Linear(784, 256))  # 命名层
model.add_module("act", nn.ReLU())
model.add_module("fc2", nn.Linear(256, 10))
```

#### 3. 通过字典或有序字典构建
```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 256)),
    ("act", nn.ReLU()),
    ("fc2", nn.Linear(256, 10))
]))
```

---

### 三、输入输出流程
#### 1. 前向传播规则
- **数据流动**：输入依次通过每一层，输出作为下一层的输入。
- **数学表达**：  
  $$
  \text{输出} = \text{layer}_N( \cdots \text{layer}_1( \text{layer}_0( \text{输入} )) )
  $$

#### 2. 示例计算
```python
input = torch.randn(32, 784)  # 输入形状 (32,784)
output = model(input)         # 输出形状 (32,10)
```

---

### 四、高级用法
#### 1. 嵌套使用Sequential
```python
encoder = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.MaxPool2d(2),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.ConvTranspose2d(64, 3, 3),
    nn.Sigmoid()
)

full_model = nn.Sequential(
    encoder,
    decoder
)
```

#### 2. 混合使用其他容器
```python
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self.path(x)  # 残差连接

model = nn.Sequential(
    ResidualBlock(),
    ResidualBlock()
)
```

#### 3. 动态选择路径
```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    # 根据条件选择不同分支（需自定义）
    nn.Sequential(nn.Linear(20,10)) if use_simple_head 
       else nn.Sequential(nn.Linear(20,30), nn.Linear(30,10))
)
```

---

### 五、应用场景
| 场景             | 示例                     | 优势         |
| ---------------- | ------------------------ | ------------ |
| **快速原型开发** | 线性堆叠的CNN/MLP        | 减少样板代码 |
| **子模块封装**   | 将重复结构封装为Block    | 提升复用性   |
| **迁移学习**     | 替换预训练模型的最后几层 | 便捷结构调整 |

---

### 六、常见问题及解决
#### 1. 层未按预期顺序执行
- **错误示例**：  
  ```python
  # 错误：未用逗号分隔层，导致层被合并计算
  model = nn.Sequential(
      nn.Linear(10, 20) 
      nn.ReLU()  # ❌ 缺少逗号
  )
  ```
- **修正**：确保每层用逗号分隔。

#### 2. 参数未正确注册
- **错误代码**：  
  ```python
  model = nn.Sequential()
  model.my_layer = nn.Linear(10, 20)  # ❌ 不会被注册
  ```
- **修正**：使用 `add_module()` 或直接传入构造函数。

#### 3. 无法处理分支结构
- **局限性**：Sequential只能处理线性流，无法实现条件分支或并行计算。  
- **替代方案**：自定义 `nn.Module` 并重写 `forward()`。

---

### 七、与 `nn.ModuleList` 对比
| 特性         | `nn.Sequential` | `nn.ModuleList`    |
| ------------ | --------------- | ------------------ |
| **前向传播** | 自动按顺序执行  | 需手动定义执行逻辑 |
| **访问层**   | 通过索引或名称  | 通过列表索引       |
| **适用场景** | 线性结构        | 动态层集合         |

---

### 八、最佳实践
1. **命名规范**  
   为关键层添加有意义的名称，方便调试：
   ```python
   model = nn.Sequential(OrderedDict([
       ("conv1", nn.Conv2d(3, 64, 3)),
       ("pool1", nn.MaxPool2d(2)),
       ("act1", nn.ReLU())
   ]))
   ```

2. **配合类型检查**  
   在复杂模型中验证层类型：
   ```python
   assert isinstance(model[0], nn.Conv2d)
   ```

3. **可视化工具集成**  
   **使用TensorBoard可视化Sequential结构：**
   
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter()
   writer.add_graph(model, torch.randn(1, 784))
   ```

---

### 九、总结
`nn.Sequential` 的核心价值：
- **代码简洁性**：一行代码定义多级网络。
- **结构透明性**：层间关系一目了然。
- **维护便利性**：轻松调整层顺序或替换组件。

**适用场景**：  
- 超过90%的CNN/MLP基础结构  
- 需要快速验证的模型原型  
- 作为复杂模型的子模块  

**不适用场景**：  
- 需要条件分支（如ResNet的残差连接）  
- 动态计算图（如循环结构）  
- 并行处理多路径输入  

合理使用 `nn.Sequential` 可以大幅提升开发效率，但在复杂模型中仍需结合自定义 `nn.Module` 实现灵活控制。