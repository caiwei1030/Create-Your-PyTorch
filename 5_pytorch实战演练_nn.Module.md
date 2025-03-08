# nn.Module

---

`nn.Module` 是 PyTorch 中构建神经网络模型的**基石类**，所有自定义模型必须继承它。

它提供了参数管理、模型结构定义、设备切换、序列化等核心功能。

**示例：**

```
from torch import nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,iput):
        output=iput+1
        return output

model=MyModel()
z=torch.tensor(1.0)
output=model(z)

print(output)
```



---

### 一、核心作用
#### 1. **参数自动管理**
   - **自动追踪参数**：所有通过 `nn.Parameter` 定义的参数会被自动注册，可通过 `parameters()` 方法访问。
   - **示例**：
     ```python
     class MyModel(nn.Module):
         def __init__(self):
             super().__init__()
             self.weight = nn.Parameter(torch.randn(3, 5))  # 自动追踪参数
     ```
   - **查看参数**：
     ```python
     model = MyModel()
     for name, param in model.named_parameters():
         print(name, param.shape)  # 输出：weight torch.Size([3, 5])
     ```

#### 2. **模型结构组织**
   - **子模块嵌套**：可通过嵌套 `nn.Module` 构建复杂网络（如层、块、子网络）。
   - **示例**：
     ```python
     class SubBlock(nn.Module):
         def __init__(self):
             super().__init__()
             self.conv = nn.Conv2d(3, 64, 3)
     
     class MyModel(nn.Module):
         def __init__(self):
             super().__init__()
             self.block = SubBlock()  # 子模块
     ```

#### 3. **前向传播定义**
   - **必须实现 `forward()`**：定义数据如何通过网络流动。
   - **示例**：
     ```python
     class MyModel(nn.Module):
         def __init__(self):
             super().__init__()
             self.fc = nn.Linear(10, 2)
     
         def forward(self, x):
             return self.fc(x)
     ```

#### 4. **设备与数据类型管理**
   - **自动切换设备**：使用 `to(device)` 将模型参数和缓冲区移动到 GPU 或 CPU。
   - **示例**：
     ```python
     device = "cuda" if torch.cuda.is_available() else "cpu"
     model = MyModel().to(device)  # 所有参数转移到指定设备
     ```

---

### 二、核心方法详解
#### 1. **参数管理**
   - **访问参数**：
     ```python
     # 获取所有参数（生成器）
     params = list(model.parameters())
     
     # 获取参数名和值
     for name, param in model.named_parameters():
         print(name, param.shape)
     ```

   - **参数初始化**：
     ```python
     def init_weights(m):
         if isinstance(m, nn.Linear):
             nn.init.xavier_normal_(m.weight)
     
     model.apply(init_weights)  # 递归初始化所有子模块
     ```

#### 2. **模型保存与加载**
   - **保存模型**：
     ```python
     torch.save(model.state_dict(), "model.pth")  # 仅保存参数
     ```

   - **加载模型**：
     ```python
     model = MyModel()
     model.load_state_dict(torch.load("model.pth"))
     ```

#### 3. **钩子函数（Hooks）**
   - **监控中间层输出**：
     ```python
     def hook_fn(module, input, output):
         print(f"Layer {module.__class__.__name__} 输出形状: {output.shape}")
     
     # 注册钩子
     handle = model.fc.register_forward_hook(hook_fn)
     
     # 移除钩子
     handle.remove()
     ```

#### 4. **模型模式切换**
   - **训练模式**：`model.train()` （启用 `Dropout`、`BatchNorm` 的训练行为）。
   - **评估模式**：`model.eval()` （关闭随机性，用于推理）。

---

### 三、代码示例：构建一个简单CNN
```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)    # 输入通道3，输出通道16，卷积核3x3
        self.pool = nn.MaxPool2d(2, 2)      # 池化层
        self.fc = nn.Linear(16 * 14 * 14, 10)  # 全连接层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 → ReLU → 池化
        x = x.view(-1, 16 * 14 * 14)         # 展平
        x = self.fc(x)
        return x

# 使用示例
model = CNN()
print(model)
```

---

### 四、常见错误及解决
#### 1. **忘记调用 `super().__init__()`**
   - **错误现象**：参数未被正确追踪。
   - **修正**：在子类 `__init__` 中必须首先调用 `super().__init__()`。

#### 2. **在 `forward` 中使用 Tensor 操作而非模块**
   - **错误示例**：
     ```python
     def forward(self, x):
         x = x.view(x.size(0), -1)  # 正确：使用张量操作
         x = torch.relu(x)          # 错误：应使用 nn.ReLU() 以允许参数追踪
     ```
   - **修正**：若需自定义操作，使用 `nn.Module` 封装或确保不涉及参数。

#### 3. **设备不匹配**
   - **错误现象**：输入数据在 CPU，模型参数在 GPU。
   - **修正**：统一设备和数据类型：
     ```python
     model = model.to(device)
     inputs = inputs.to(device)
     ```

---

### 五、高级用法
#### 1. **自定义层**
   ```python
   class MyLayer(nn.Module):
       def __init__(self, in_dim, out_dim):
           super().__init__()
           self.weights = nn.Parameter(torch.randn(in_dim, out_dim))
       
       def forward(self, x):
           return x @ self.weights

   model = nn.Sequential(MyLayer(10, 5), nn.ReLU())
   ```

#### 2. **动态计算图**
   - **在 `forward` 中使用 Python 控制流**：
     ```python
     def forward(self, x):
         if x.sum() > 0:
             return self.path_a(x)
         else:
             return self.path_b(x)
     ```

#### 3. **混合精度训练**
   - **自动转换数据类型**：
     ```python
     from torch.cuda.amp import autocast
     
     with autocast():
         outputs = model(inputs)
     ```

---

### 六、总结
`nn.Module` 的核心作用：
- **结构化网络设计**：通过继承和组合构建复杂模型。
- **自动化参数管理**：自动追踪梯度、设备切换。
- **模块化扩展**：支持自定义层、钩子、动态计算图。
- **便捷部署**：提供模型保存、加载和导出接口。

掌握 `nn.Module` 的使用是 PyTorch 模型开发的基础，合理利用其特性可以大幅提升开发效率和代码可维护性。