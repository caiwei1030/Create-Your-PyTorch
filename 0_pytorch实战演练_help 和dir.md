# help 和dir

> 按住CTRL键，可以进入创建函数的类里面，有详细用法。

① Python3.6.3相当于一个package，package里面有不同的区域，不同的区域有不同的工具。

② Python语法有两大法宝：dir()、help() 函数。

```
import torch

print(torch.cuda.is_available())
help(torch.cuda.is_available())
dir(torch)
```

- dir()：打开，看见里面有多少分区、多少工具。
- help()：说明书。

**cuda.is_available()的用法**

```
def is_available() -> bool:
    r"""Return a bool indicating if CUDA is currently available."""
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0
```

> [!NOTE]
>
> """Return a bool indicating if CUDA is currently available."""

