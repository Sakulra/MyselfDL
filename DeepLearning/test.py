# import d2l
# import numpy as np
# import pandas as pd

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
###################################向tenser矩阵传递向量索引
# import torch
# index=torch.tensor([0,2])
# a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(a[index])
#################
from torch.utils import data
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66])

# TensorDataset对tensor进行打包
train_ids = data.TensorDataset(a, b) 
for x_train, y_label in train_ids:
    print(x_train, y_label)

# dataloader进行数据封装
print('=' * 80)
train_loader = data.DataLoader(dataset=train_ids, batch_size=3, shuffle=True)
for i, data in enumerate(train_loader, 1):  
# 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))