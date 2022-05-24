"""
这个文件没有用，中间测试代码
"""
import torch

x = torch.linspace(1, 100, 100)       # x data (torch tensor)
y = torch.linspace(100, 1, 100)       # y data (torch tensor)

x = x.reshape((-1, 2, 5))
x = torch.transpose(x, 0, 1)
print(x)
print(x.shape)