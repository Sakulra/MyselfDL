import torch
from torch import nn


#允许我们在不存在所需所有GPU的情况下运行代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')#f''格式化字符串
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


#查询张量所在设备,创建张量默认在cpu上
x = torch.tensor([1,2,3])
print(x.device)
#在gpu上创建
X = torch.ones(2,3,device=try_gpu())#在第0个gpu上创建
Y = torch.rand(2,3,device=try_gpu(1))
print(X)

#不同GPU上的计算，需要挪到同一个GPU上才能进行计算
Z = X.cuda(1)#Y在GPU1，X在GPU0，所以借助中间变量Z，将X复制过去
print(X)
print(Z)
print(Y + Z)#现在Y和Z都在1上
#如果Z已经在1上那么执行
Z.cude(1)#并不会再copy，而是返回它自己

#在GPU上做神经网络的计算
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
#确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)