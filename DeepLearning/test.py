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
# c=torch.tensor([1])
# print(a+c)

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
#######################################模型网络参数形状##########################################################
# import torch.nn as nn
# import torch
# @torch.no_grad()
# def init_weights(m):
#     print(m)
#     if type(m) == nn.Linear:
#         m.weight.fill_(1.0)
#         print(m.weight)
# net = nn.Sequential(nn.Linear(4,2,bias=False), nn.Linear(2, 3,bias=True))
# print(net)
# print('isinstance torch.nn.Module',isinstance(net,torch.nn.Module))
# print(' ')
# net.apply(init_weights)
############################################################################################################
# import pandas as pd

# data1 = pd.DataFrame({'age':[1.0,1.0,3.0,4.0], 'goal':[91,91,93,94]})
# print(data1)
# idx = data1.dtypes.index
# data1[idx]=data1[idx].apply(
#     #lambda x: (x-x.mean())/x.std()
#     lambda x:print(x+x)
# )
# #print(data1)

############################################ K 折获取训练集和测试集##############################################################
# import torch
# def get_k_fold_data(k, i, X, y):
#     assert k > 1
#     fold_size = X.shape[0] // k
#     valid_start = i * fold_size 
#     valid_end = (i + 1) * fold_size 

#     #把训练集分割出来
#     X_head, y_head = X[:valid_start, :], y[:valid_start]
#     X_tail, y_tail = X[valid_end:, :], y[valid_end:]
#     #这个判断想不出来
#     if i == 0:
#         valid_start = None      
#         X_head, y_head = torch.tensor([]), torch.tensor([])
#     elif i == k - 1:
#         valid_end = None        
#         X_tail, y_tail = torch.tensor([]), torch.tensor([])
        
#     #把测试集分割出来
#     valid_idx = slice(valid_start, valid_end)
#     X_valid, y_valid = X[valid_idx, :], y[valid_idx]
#     #把训练集连接起来
#     X_train = torch.cat([X_head, X_tail], 0)
#     y_train = torch.cat([y_head, y_tail], 0)    
#     return X_train, y_train, X_valid, y_valid
######################################################################################################
# import pandas as pd
# import numpy as np
# data=pd.DataFrame(np.arange(9).reshape(3,3),index=list('abc'),columns=list('ABC'))
# print(type(data.values))
# ts = data.iloc[:,:-1]
# print(ts)
############################################层和块###########################################################
#平行块：它以两个块为参数，例如net1和net2，并返回前向传播中两个网络的串联输出5.1.6
import scipy.io

mat_file = scipy.io.loadmat('D:\\shiyan_data\\cft\\12k2024年7月8日9时25分25.mat')

#print(mat_file.keys())
print(mat_file['T'])
