# transforms

> transforms是torchvision库中的一个模块，主要用于图像的预处理和数据增强。
>
> 常见的操作包括调整大小、裁剪、归一化、转换为张量等等。

## 一、transforms大观

![image-20250305213656131](C:\Users\br\AppData\Roaming\Typora\typora-user-images\image-20250305213656131.png)

- transforms可以看作一个工具箱，其中的一些class类可以看作成为一些工具
- 现在就是从工具箱中取出工具创建实例，用实例进行transforms

![image-20250305222442464](C:\Users\br\AppData\Roaming\Typora\typora-user-images\image-20250305222442464.png)

## 二、**`transforms.ToTensor()`**

将PIL图像或`numpy.ndarray`转换为`torch.Tensor`，并自动将像素值从`[0, 255]`缩放到`[0, 1]`。

```
from PIL import Image

img = Image.open("image.jpg")
tensor_img = transforms.ToTensor((img))  # 输出形状为 [C, H, W]
```

**注意事项**：

- 输入必须是`PIL.Image`或`numpy.ndarray`（形状为`H×W×C`）。
- 转换后张量形状为`[通道, 高度, 宽度]`。

> [!NOTE]
>
> **为何要使用tensor数据类型？**
>
> Tensor有一些属性，比如反向传播、梯度等属性，它包装了神经网络需要的一些属性。

## 三、常见的transforms工具

>  Transforms的工具主要关注他的输入、输出、作用。

### 0、__call__魔术方法使用

```
class Person:
    def __call__(self,name):
        print("__call__ "+"Hello "+name)
        
    def hello(self,name):
        print("hello "+name)
        
person = Person()  # 实例化对象
person("zhangsan") # 调用__call__魔术方法
person.hello("list") # 调用hello方法
```

输出结果：

```
__call__ Hello zhangsan
hello list
```

### **1、`transforms.Normalize()`**

**功能**：对张量图像进行归一化（需在`ToTensor()`之后使用）。

**参数**：

- `mean` (list): 各通道的均值。
- `std` (list): 各通道的标准差。

**示例**：

```
# 输入为 [C, H, W] 的张量
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_img = transform(tensor_img)  # 输出像素范围可能为负值：
```



###  **2、`transforms.Resize()`**

**功能**：调整图像尺寸。

**参数**：

- `size` (int或tuple):
  - 若为整数，将图像的短边缩放到该值，长边按比例缩放。
  - 若为`(height, width)`，直接缩放到指定尺寸。
- `interpolation` (InterpolationMode): 插值方法（默认为`BILINEAR`）。

**示例**：

```
transform = transforms.Resize((224, 224))  # 强制缩放到224x224
resized_img = transform(img)
```

### **3、`transforms.Compose()`**

**功能**：将多个图像变换组合成一个顺序执行的流水线。

**参数**：

- `transforms` (list): 由多个`transforms`操作组成的列表，按顺序执行。

**示例**：

```
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**注意事项**：

- 顺序至关重要，通常先调整尺寸/裁剪，再转为张量，最后归一化。

> [!CAUTION]
>
> 注意Compose（）里面的参数是由列表组成的



## 四、总结

- **关注输入输出类型**
- **多看官方文档**
- **关注方法需要说明参数**

> [!NOTE]
>
> **在不知道返回什么值时候**
>
> - **print**
> - **print（type())**
> - **debug**

