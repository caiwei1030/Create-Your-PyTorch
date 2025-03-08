# Dataset

在PyTorch中，`Dataset` 类用于封装数据，提供统一的访问接口，使数据加载更高效且模块化。

它需要用户继承并实现两个核心方法：

`__len__`（返回数据集大小）和 `__getitem__`（根据索引返回样本）。

通过结合 `DataLoader`，可以实现批量加载、多进程读取等功能。

**实际上就是定义组织数据的规范。**

## **1、导入库**

```
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
```

**Dataset的作用**

1. **数据封装**：将数据（如特征、标签）组织成结构化的形式。
2. **按需加载**：支持索引访问，避免一次性加载全部数据到内存。
3. **数据预处理**：在 `__getitem__` 中实现数据增强（如裁剪、归一化）。
4. **与 `DataLoader` 协作**：自动分批、打乱顺序、多线程加速。



## 2、核心方法

**1. `__init__` 方法**

**功能**

- **初始化数据源**：加载数据路径、读取元数据（如 CSV 文件）、设置预处理参数等。
- **内存管理**：如果数据量较小，可直接将数据加载到内存；如果数据量大，仅保存索引或文件路径。
- **参数配置**：定义数据增强、归一化等操作的参数（如是否启用随机裁剪）。

**实现方式**

- 接收数据路径或原始数据（如张量、列表）。
- 解析数据并存储到成员变量（如 `self.data`、`self.labels`）。
- 初始化预处理工具（如 `transforms`）

**2. `__len__` 方法**

**功能**

- **返回数据集大小**：告诉 `DataLoader` 总共有多少个样本。
- **支持动态数据集**：如果数据是动态生成的（如在线生成），需在此返回逻辑上的样本总数。

**实现方式**

- 直接返回存储的数据长度（如 `len(self.image_paths)`）。

**3. `__getitem__` 方法**

**功能**

- **按索引返回样本**：根据 `index` 加载数据（如从磁盘读取图像），并进行预处理。
- **数据转换**：将原始数据（如 PIL 图像）转换为张量，应用数据增强。
- **异常处理**：确保索引有效，处理文件损坏等错误。

**实现方式**

- 通过 `index` 获取单个样本的原始数据（如从 `self.image_paths[index]` 读取文件）。
- 对数据进行预处理（如归一化、随机增强）。
- 返回格式统一的数据（如 `(tensor_image, tensor_label)`）。

## 3、示例

```
class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self,idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

```

