# 网络模型的保存和读取

---

### 一、模型加载与初始化

#### 1. **加载预训练模型**
```python
import torchvision.models as models

# 加载预训练模型（自动下载权重）
model = models.resnet18(pretrained=True)

# 仅加载模型结构（不加载权重）
model = models.resnet18(pretrained=False)
```

**说明**：
- 使用`torchvision.models`提供的主流模型（如ResNet、VGG、AlexNet）
- `pretrained=True`自动下载ImageNet预训练权重
- 适合快速实现迁移学习

> [!WARNING]
>
> **pretrained在未来可能不能使用，改用**
>
> ```
> @handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
> ```



---

#### 2. **自定义初始化**
```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)  # 递归初始化所有层
```

**作用**：
- 解决梯度消失/爆炸问题
- 加速模型收敛
- 常用初始化方法：Kaiming（ReLU）、Xavier（Tanh）

---

### 二、模型结构调整

#### 1. **修改输出层（分类任务适配）**
```python
# 原模型输出1000类（ImageNet），改为10类
num_classes = 10

# 方案1：直接替换全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 方案2：构建多层分类头
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
```

**验证输出维度**：
```python
test_input = torch.randn(1, 3, 224, 224)
print(model(test_input).shape)  # 应输出torch.Size([1, 10])
```

---

#### 2. **修改中间特征层**
```python
# 替换第一个卷积层（输入通道改为1，适配灰度图）
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

# 插入注意力模块到layer3之后
class ChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//16),
            nn.ReLU(),
            nn.Linear(in_ch//16, in_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

model.layer3.add_module("channel_attn", ChannelAttention(256))
```

---

### 三、参数管理策略

#### 1. **冻结部分层（迁移学习）**
```python
# 冻结所有卷积层
for param in model.parameters():
    param.requires_grad = False

# 仅解冻全连接层
for param in model.fc.parameters():
    param.requires_grad = True

# 统计可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数：{trainable_params/1e6:.2f}M")
```

---

#### 2. **分层设置学习率**
```python
optimizer = torch.optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 1e-4},  # 底层小学习率
    {'params': model.layer1.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2}       # 顶层大学习率
], momentum=0.9, weight_decay=1e-4)
```

---

### 四、模型保存与加载

#### 1. **保存完整模型**
```python
torch.save(model, 'model.pth')  # 包含结构和参数
loaded_model = torch.load('model.pth')
```

#### 2. **保存状态字典（推荐）**
```python
torch.save(model.state_dict(), 'model_weights.pth')

# 加载时需先构建相同结构
new_model = models.resnet18()
new_model.load_state_dict(torch.load('model_weights.pth'))
```

---

### 五、模型调试与可视化

#### 1. **结构可视化**
```python
from torchsummary import summary

summary(model.cuda(), (3, 224, 224))  # 显示各层参数信息

# 输出示例：
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          ...
```

---

#### 2. **特征图可视化**
```python
# 注册前向钩子捕获特征
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer2[1].conv2.register_forward_hook(get_activation('feat_layer'))

# 前向传播后查看
_ = model(test_input)
plt.imshow(activations['feat_layer'][0, 0].cpu().numpy())
```

---

### 六、高级修改技巧

#### 1. **动态网络结构**
```python
class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(784, 256),
            nn.Linear(256, 128)
        ])
        self.drop_rate = 0.5
    
    def forward(self, x):
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i < len(self.blocks) - 1:
                x = F.relu(x)
                if self.training:  # 仅在训练时dropout
                    x = F.dropout(x, p=self.drop_rate)
        return x
```

---

#### 2. **多分支结构**
```python
class MultiPathNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_a = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.MaxPool2d(2)
        )
        self.path_b = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.AvgPool2d(2)
        )
        self.fusion = nn.Conv2d(128, 256, 1)
    
    def forward(self, x):
        x1 = self.path_a(x)
        x2 = self.path_b(x)
        return self.fusion(torch.cat([x1, x2], dim=1))
```

---

### 七、常见问题解决方案

| **问题现象**   | **可能原因**             | **解决方案**                                 |
| -------------- | ------------------------ | -------------------------------------------- |
| 输出维度错误   | 结构调整后未适配输入尺寸 | 使用`nn.AdaptiveAvgPool2d`自动调整特征图尺寸 |
| 训练时显存不足 | 模型过大或批量太大       | 减小`batch_size`，使用混合精度训练           |
| 梯度消失/爆炸  | 初始化不当或网络过深     | 使用残差连接，合理初始化权重                 |
| 验证集性能差   | 过拟合                   | 增加`Dropout`层，添加L2正则化                |

---

### 八、最佳实践建议

1. **模块化设计**  
   ```python
   class ResBlock(nn.Module):
       def __init__(self, in_ch):
           super().__init__()
           self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
           self.bn1 = nn.BatchNorm2d(in_ch)
           self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
           self.bn2 = nn.BatchNorm2d(in_ch)
       
       def forward(self, x):
           residual = x
           x = F.relu(self.bn1(self.conv1(x)))
           x = self.bn2(self.conv2(x))
           return F.relu(x + residual)
   ```

2. **版本控制**  
   ```python
   # 保存检查点时包含超参数
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, 'checkpoint.pth')
   ```

3. **设备兼容性**  
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

---

通过以上方法，您可以：
- **快速复用**现有模型架构
- **灵活调整**模型结构适应新任务
- **有效管理**模型参数和训练过程
- **深度优化**模型性能与效率

建议结合具体任务需求，选择合适的修改策略，并通过实验验证修改效果。