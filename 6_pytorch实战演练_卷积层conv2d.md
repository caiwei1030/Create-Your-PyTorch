# 卷积层conv2d

```
import torch
import torchvision.datasets
import torchvision.transforms
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


train_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

model=MyModel()

writer=SummaryWriter("logs")
step=0
for data in train_loader:
    imgs,targets=data
    output=model(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("data_conv2d",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output_conv2d",output,step)
    step+=1

writer.close()
```

`nn.Conv2d` 是 PyTorch 中实现二维卷积的核心层，广泛用于图像处理和计算机视觉任务。以下是其详细用法和作用解析：

---

### 一、核心作用
1. **特征提取**  
   通过滑动卷积核（kernel）在输入数据上提取局部特征（如边缘、纹理）。
2. **空间维度压缩**  
   通过调整步长（`stride`）和填充（`padding`）控制输出特征图尺寸。
3. **通道维度变换**  
   改变输入数据的通道数（如将 RGB 三通道转换为更高维特征空间）。

---

### 二、参数详解
#### 1. 基本参数
| 参数           | 描述                       | 示例值              | 默认值 |
| -------------- | -------------------------- | ------------------- | ------ |
| `in_channels`  | 输入通道数                 | 3（RGB图像）        | 必填   |
| `out_channels` | 输出通道数（卷积核数量）   | 64                  | 必填   |
| `kernel_size`  | 卷积核尺寸                 | 3 或 (3,5)          | 必填   |
| `stride`       | 滑动步长                   | 2 或 (2,1)          | 1      |
| `padding`      | 边缘填充像素数             | 1 或 (1,2)          | 0      |
| `dilation`     | 卷积核膨胀系数（空洞卷积） | 2                   | 1      |
| `groups`       | 分组卷积的组数             | 2（深度可分离卷积） | 1      |
| `bias`         | 是否添加偏置项             | `True`/`False`      | `True` |

#### 2. 参数选择策略
- **`kernel_size`**：常用 3x3（平衡感受野和计算量）、1x1（通道维度变换）。
- **`stride`**：步长越大，输出尺寸越小（如 `stride=2` 尺寸减半）。
- **`padding`**：设置为 `kernel_size//2` 可保持输入输出尺寸相同（需配合 `stride=1`）。
- **`groups`**：  
  - `groups=1`：标准卷积（所有输入通道参与计算）。  
  - `groups=in_channels`：深度可分离卷积（MobileNet 使用）。

---

### 三、输入输出形状
#### 1. 输入张量形状
- **格式**：`(Batch_size, in_channels, Height, Width)`  
  示例：`(32, 3, 224, 224)` 表示批次大小32，3通道，224x224图像。

#### 2. 输出张量形状
- **计算公式**：  
  ```
  H_out = floor[(H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1]
  W_out = floor[(W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1]
  ```
- **简化公式**（当 `dilation=1` 时）：  
  ```
  H_out = (H_in + 2*padding - kernel_size) // stride + 1
  W_out = (W_in + 2*padding - kernel_size) // stride + 1
  ```

#### 3. 示例计算
```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
input = torch.randn(32, 3, 224, 224)  # 输入形状 (32,3,224,224)
output = conv(input)
print(output.shape)  # 输出形状 (32,64,112,112)
```

---

### 四、代码示例：构建卷积块
```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 保持尺寸不变
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 尺寸减半
        )
    
    def forward(self, x):
        return self.conv(x)

# 使用示例
block = ConvBlock(3, 64)
input = torch.randn(1, 3, 224, 224)
output = block(input)
print(output.shape)  # 输出形状 (1,64,112,112)
```

---

### 五、常见应用场景
#### 1. 图像分类（如 ResNet）
```python
self.features = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.MaxPool2d(3, stride=2)
)
```

#### 2. 目标检测（如 YOLO）
```python
# Darknet-53 中的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, 1)
        self.conv2 = nn.Conv2d(in_ch//2, in_ch, 3, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))
```

#### 3. 图像分割（如 U-Net）
```python
# 下采样模块
self.down = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

---

### 六、常见问题及解决
#### 1. 输入形状错误
- **错误信息**：`RuntimeError: Given input size: ...`  
- **原因**：输入尺寸不符合卷积参数（如 `kernel_size > input_size`）。  
- **解决**：  
  - 增加 `padding`  
  - 调整 `stride`  
  - 使用 `nn.LazyConv2d`（自动推断 `in_channels`）

#### 2. 显存不足
- **现象**：训练时出现 CUDA out of memory。  
- **优化策略**：  
  - 减小 `batch_size`  
  - 使用更小的 `kernel_size` 或 `out_channels`  
  - 使用深度可分离卷积（`groups=in_channels`）

#### 3. 梯度消失/爆炸
- **表现**：模型无法收敛。  
- **解决方案**：  
  - 使用 `nn.init.kaiming_normal_` 初始化权重  
  - 添加 Batch Normalization 层  
  - 降低学习率

---

### 七、高级技巧
#### 1. 空洞卷积（Dilated Convolution）
```python
# 扩大感受野而不增加参数量
conv = nn.Conv2d(64, 128, kernel_size=3, dilation=2)
```

#### 2. 分组卷积（Grouped Convolution）
```python
# 减少计算量（用于MobileNet、ResNeXt）
conv = nn.Conv2d(64, 128, kernel_size=3, groups=4)
```

#### 3. 动态卷积（Dynamic Convolution）
```python
# 根据输入动态生成卷积核（如 CondConv）
# 需自定义实现，此处为示意代码
class DynamicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight_gen = nn.Linear(in_ch, out_ch*in_ch*3*3)
    
    def forward(self, x):
        batch_size = x.size(0)
        weights = self.weight_gen(x.mean(dim=[2,3])).view(batch_size, -1, 3, 3)
        return F.conv2d(x, weights, padding=1)
```

---

### 八、总结
`nn.Conv2d` 的核心价值：
- **局部感知**：通过卷积核捕捉空间局部特征。
- **参数共享**：同一卷积核在整个输入上滑动，减少参数量。
- **维度控制**：灵活调整输出通道数和特征图尺寸。

合理选择卷积参数（如 `kernel_size`、`stride`、`padding`）是构建高效 CNN 模型的关键。结合 BatchNorm、激活函数和池化层，可以构建出强大的特征提取器。