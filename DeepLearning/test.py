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
##################################################################################################
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

###########################################
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

import torch
a=torch.tensor([[1,2,3],[4,5,6]])
b=torch.tensor([1,1,1])
print(a+b)