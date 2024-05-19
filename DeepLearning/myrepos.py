import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch.nn as nn

def set_axes(axes, xlabel, ylabel, xlim=None, ylim=None, xscale='linear', yscale='linear', legend=None):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    #在图形中添加网格线
    axes.grid()

################定义类Timer#############使用循环和使用按元素相加运行时间对比#############
# n=10000
# a=torch.ones(n)
# b=torch.ones(n)

class Timer:
    def __init__(self) -> None:
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
# c=torch.zeros(n)
# timer=Timer()
# for i in range(10000):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec(秒)')

# timer.start()
# d = a + b
# print(f'{timer.stop():.5f} sec(秒)')
##########################################不同方差和均值的正态分布可视化####################
# def normal(x,miu,sigma):
#     p=1/math.sqrt(2*math.pi*sigma**2)
#     return p*np.exp(-0.5/sigma**2*(x-miu)**2)
# x = np.arange(-7,7,0.01)
# params = [(0,1),(0,2),(3,1)]

# params = [(0, 1), (0, 2), (3, 1)]
# d2l.plot(x, [normal(x, miu, sigma) for miu, sigma in params], xlabel='x',
#          ylabel='p(x)', figsize=(4.5, 2.5),
#          legend=[f'mean {miu}, std {sigma}' for miu, sigma in params])
# plt.show()
# # plt.scatter()#绘制散点图
#####################################################################
#定义一个实用程序类Accumulator，用于对多个变量进行累加
class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n) -> None:#n是待累加的变量的个数
        self.data = [0.0]*n#有几个数，就把列表扩成相应大小,data 用于存储待累加的变量，如正确率、损失值，样例数等
        #如果不把data初始化为n个0就无法进行下一步的累加了，data列表是空的话，怎么让data的内容与传入的args的内容进行加法

    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]
        #对于index i 从data中取出a，从args中取出b，然后相加，结果放在data 的i索引位置

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
#####################################激活函数############################################
#relu
# #relu激活函数,被定义为输入元素与0的最大值：
#x = torch.arange(-8.0,8.0,0.1,requires_grad=True)

#y = torch.relu(x)
# print(x.shape==y.shape)
# figs,axs = plt.subplots(2,1,figsize=(5,2.5),sharey=True,sharex=True)
# axs[0].plot(x.detach(),y.detach())
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('relu(x)')

# #torch.one_like(x)表示创建一个跟x同型的全一矩阵作为梯度的初始值
# y.backward(torch.ones_like(x),retain_graph=True)
# axs[1].plot(x.detach(),x.grad)
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('grand of relu(x)')
# plt.show()

#sigmoid
# y = torch.sigmoid(x)
# figs,axs = d2l.plt.subplots(2,1,sharex=True,sharey=True)
# #d2l.plot(x.detach(),y.detach(),'x','sigmoid(x)')
# axs[0].plot(x.detach(),y.detach())
# set_axes(axs[0],'x','sigmoid(x)')

# #x.grad.data.zero_()
# y.backward(torch.ones_like(x),retain_graph=True)
# #d2l.plot(x.detach(),x.grad,'x','grad of sigmoid')
# axs[1].plot(x.detach(),x.grad)
# set_axes(axs[1],'x','grad of sigmoid')
# plt.show()

#tanh激活函数
# y = torch.tanh(x)
# fig,axs = d2l.plt.subplots(2,1)
# axs[0].plot(x.detach(),y.detach())
# set_axes(axs[0],'x','tanh(x)')

# y.backward(torch.ones_like(x),retain_graph=True)
# axs[1].plot(x.detach(),x.grad)
# set_axes(axs[1],'x','grad of tanh(x)')
# d2l.plt.show()

############################################################################
# @torch.no_grad()
# def init_weights(m):
#     print(m)
    
# net = nn.Sequential(nn.Linear(2,4), nn.Linear(4, 8))
# print(net)
# print('isinstance torch.nn.Module',isinstance(net,torch.nn.Module))
# print(' ')
# net.apply(init_weights)

# @torch.no_grad()
# def init_weights(m):
#     print(m)
#     if type(m) == nn.Linear:
#         m.weight.fill_(1.0)
#         print(m.weight)
# net = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 8))
# net.apply(init_weights)

#################################################梯度爆炸和消失###############################################
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.show()

M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)