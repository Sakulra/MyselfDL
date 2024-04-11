# import d2l
# import numpy as np
# import pandas as pd
# import torch

# a=pd.DataFrame(np.arange(9).reshape(3,-1),index=['a','b','c'],columns=['A','B','C'])

# print(a,a.iloc[1,1])

# import os

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
###################################向tenser矩阵传递向量索引#########################################
# import torch
# index=torch.tensor([0,2])
# a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(a[index])
##################################数据集封装和打包################################################################
# from torch.utils import data
# import torch

# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66])

# TensorDataset对tensor进行打包
# train_ids = data.TensorDataset(a, b) 
# print(train_ids[1])
# for x_train, y_label in train_ids:
#     print(x_train, y_label)

# dataloader进行数据封装
# print('=' * 80)
# train_loader = data.DataLoader(dataset=train_ids, batch_size=3, shuffle=True)
# print(train_loader)
# for data in train_loader:
#     print(data)
# for i, data in enumerate(train_loader, 1):  
# # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
#     x_data, label = data
#     print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))

#######################################################################################3
#import torch
# a = torch.tensor([[1,2,3]],[4,5,6])
# b = torch.tensor([1,2,3])
# print(a+b)

# x = torch.arange(12).reshape(4,3)
# print(x)
# rows = torch.tensor([[0,0],[3,3]]) 
# cols = torch.tensor([[0,2],[0,2]])
# y = x[rows,cols]
# print (y)

# y_hat = torch.arange(6).reshape(2,3)
# print(y_hat)
# y = torch.tensor([0,2])
# print(range(len(y_hat)))
# print (y_hat[range(len(y_hat)),y])

# list1 = [1.0,2.0,3.0]
# b = [1,2,3]
# list1 = [i+j for i,j in zip(list1,b)]
# print(list1)
# list2 = [9,]
# list2 = [i for i in b]
# print(list2)
#列表推导式原型不应该是这样的吗
# list2 = []
# for i,j in zip(list2,b):
#     list2.append(i+j)
# print(list2)
##############################################################################
# a = torch.tensor([1,2,3])
# b = torch.tensor([1,1,1])
# cmp = a==b
# print(cmp)

# from matplotlib import pyplot as plt

# fig,axes = plt.subplots(2,2,figsize=(3.5,2.5))
# print(axes[0])
#####################################################################################3
# import torch
# from torch import nn

# a=torch.tensor([[1,2,3],[4,5,6]])
# b=torch.tensor([1,1,1])
# print(a+b)

# w = torch.empty(3, 5)
# print(w)
# print('.'*10)
# print(nn.init.normal_(w))
###########################################动态画图##############################################3
#import matplotlib.pyplot as plt
# import math
# i=0
# x=[]
# y1=[]
# y2=[] # 要绘制的是（x,y1）和（x,y2）
# # subplot(在窗口中分的行、列、画图序列数)
# while (i<100):
#     plt.clf()  # 清除之前画的图
#     # subplot(在窗口中分的行、列、画图序列数)
#     plt.subplot(211) #第1个图画在一个两行一列分割图的第1幅位置
#     x.append(i)
#     y1.append(i**2)
#     plt.plot(x,y1)
#     plt.subplot(212) #第2个图画在一个两行一列分割图的第2幅位置
#     y2.append(math.sqrt(i))
#     plt.plot(x,y2)
#     plt.pause(0.01)  # 暂停0.1秒
#     plt.ioff()  # 关闭画图的窗口
#     i=i+1

# x=[]
# y=[]
# i=0
# while i<=100:
#     x.append(i)
#     y.append(i**2)
#     plt.plot(x,y)
#     plt.draw()
#     i=i+1
#     plt.pause(0.1)
################################自定义网络######################################################
# import torch
# import torch.nn as nn

# # 在torch中，只会处理2维的数据
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# # x.pow(2)的意思是x的平方
# y = x.pow(2) + 0.2 * torch.rand(x.size())


# class Net(torch.nn.Module):  # 继承torch的module
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()  # 继承__init__功能
#         # 定义每一层用什么样的样式
#         self.hidden1 = torch.nn.Linear(n_feature, n_hidden,bias= 1)  # 隐藏层线性输出
#         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden,bias=1)  # 隐藏层线性输出
#         self.predict = torch.nn.Linear(n_hidden, n_output,bias=0)  # 输出层线性输出

#     def forward(self, x):
#         # 激励函数（隐藏层的线性值）
#         x = torch.relu(self.hidden1(x))
#         x = torch.relu(self.hidden2(x))
#         x = self.predict(x)  # 输出值
#         return x


# net = Net(2, 5, 3)
# print(net)
# print(net.parameters())
# paras = list(net.parameters())
# for num,para in enumerate(paras):
#     print('number:',num)
#     print(para)
#     print('_________________________________________')
#############################################矩阵花样次方###################################################
# import numpy as np

# x=[[1],[2],[3],[4]]
# idx=[2,2]
# print(np.power(x,np.arange(3).reshape(1,-1)))

######################################################向量到底是行还是列#######################
# import torch
# matrix = torch.arange(12).reshape(3,4)
# b = torch.tensor([1,2,3,4])
# c = torch.tensor([[1],[2],[3],[4]])
# print(matrix)
# print(b)
# print(torch.mv(matrix,b))
# print('*'*10)
# print(c)
# print(torch.mm(matrix,c))
###############################################矩阵负索引#################################
# import torch
# matrix = torch.arange(12).reshape(3,4)
# print(matrix)
# print(matrix[:,-2])#可以把matrix看成一个列表，里面的元素还是列表，直接索引就是对最外面那个列表来说的索引
# print(matrix.shape[0])
################################################################################################
# def fn(arg1,arg2,arg3):
#    print("arg1:",arg1)
#    print("arg2:",arg2)
#    print("arg3:",arg3)
# args = ('two',1,2)
# kwargs = {'arg1':'one','arg2':2,'arg3':'three'}
# fn(**kwargs)
#################################################################################################
#高维线性回归
import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#生成数据y=0.05+∑0.01xi+ε，ε∈N（0，0.01**2）
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs,1))*0.01, 0.05

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

train_data = synthetic_data(true_w,true_b,n_train)
train_iter = load_array(train_data,batch_size)
test_data = synthetic_data(true_w,true_b,batch_size)
test_iter = load_array(test_data,batch_size,is_train=False)

#初始化参数模型
def init_params():
    w = torch.normal(0, 1, size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

#L2范数
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#优化函数
def updater(batch_size,w,b,lr):
    """优化函数：随机梯度下降sgd"""
    return d2l.sgd([w,b],lr,batch_size)

#训练代码
def train(lambd:"λ"):
    w, b = init_params()
    net = lambda X: torch.matmul(X,w) + b
    loss = d2l.squared_loss
    num_epoch, lr = 100, 0.03
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[5,num_epoch],legend=['train','test'])
    for epoch in range(num_epoch):
        for X, y in train_iter:
            #增加L2范数惩罚项
            #广播机制使l2_penalty变为batch_size的向量
            l = loss(net(X),y) + lambd*l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net,train_iter,loss),
                                    d2l.evaluate_loss(net,test_iter,loss)))
        print('w的L2范数是：', torch.norm(w).item())

#忽略正则化直接训练，必过拟合
#train(lambd=0)
#使用权重衰减
train(lambd=3)
d2l.plt.show()