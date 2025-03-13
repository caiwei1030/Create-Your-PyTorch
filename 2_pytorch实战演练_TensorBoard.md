# Tensorboard的使用

> 小技巧：
>
> 在pycharm中按住ctrl键点击函数可以直接访问函数的构造方法

```
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
```

## 1、SummaryWriter

```
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")
```

**构造一个实例writer，会创建一个logs的文件夹，writer写的内容全都会放在文件夹下**

**写的内容会产生一些日志文件**

> [!WARNING]
>
> **在使用完writer实例之后，一定要用writer.close()关闭实例**

常用的一些写日志的方法

**add_scalar**

```
writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
writer.close()
```

**功能**：记录单个标量数据（如损失、准确率），在TensorBoard中以折线图形式展示。

**参数说明**：

- `tag` (str): 数据标签，用于分类和图表标题。
- `scalar_value` (float): 要记录的标量值（如损失值）。
- `global_step` (int, optional): 记录的步数或epoch数（X轴坐标）。
- `walltime` (float, optional): 覆盖默认的时间戳，通常无需设置。

> [!NOTE]
>
> - 同一`tag`多次调用会生成连续曲线，不同`tag`会分开显示。
> - 可通过`global_step`参数跟踪训练进度（如batch迭代数或epoch数）。



**writer.add_image**()

**功能**：记录单张图像，适用于可视化输入数据、特征图或生成结果。

**参数说明**：

- `tag` (str): 图像标签，用于分类和显示。
- `img_tensor` (Tensor/np.ndarray): 图像数据张量，支持多种格式（需指定`dataformats`）。
- `global_step` (int, optional): 记录图像的步骤（如epoch或batch数）。
- `dataformats` (str): 输入张量的维度格式，默认为`'CHW'`（通道×高×宽）。

> [!NOTE]
>
> - 默认支持`CHW`（通道优先），若为`HWC`（高×宽×通道），需显式设置`dataformats='HWC'`。
> - 灰度图需扩展为`1xHxW`或`HxWx1`（单通道）。

> [!CAUTION]
>
> 如果要记录多张图像，应使用方法add_images

## （2）在tensor_board中显示

在pycharm的terminal中输入

```
tensorboard --logdir=<实例参数名字>
```

为避免多人使用端口导致冲突，也可以在后面加上后缀，使得端口独立

```
tensorboard --logdir=<实例参数名字> --port=6008
```

