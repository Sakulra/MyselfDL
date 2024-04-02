import numpy as np
import torch
from d2l import torch as d2l

from torch.utils import data
from torch import nn

#人工生成数据集(没变)
def synthetic_data(w,b,num_examples): #num_examples是样本数
    """生成y=Xw+b+噪声"""
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w) + b
    y+=torch.normal(0,0.1,y.shape)
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

#读取数据集(变化)
def load_array(data_arrays:'featureas and labels of dataset',batch_size,is_train=True):
    """构造一个Pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=True)

batch_size = 10#分成10批，一批一百个
data_iter = load_array((features,labels),batch_size)

#使用iter构造Python迭代器，并使用next从迭代器中获取第一项。
next(iter(data_iter))

#定义模型
#我们首先定义一个模型变量net，它是一个Sequential类的实例。Sequential类将多个层串联在一起。
#当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。
net = nn.Sequential(nn.Linear(2,1))
#nn.Linear(in,out)它有两个参数，第一个指定输入特征形状，即2，第二个指定输出特征形状，
#输出特征形状为单个标量，故为1.

#初始化模型参数
#在使用net之前，我们需要初始化模型参数。 如在线性回归模型中的权重和偏置。
#在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
#通过net[0]选择网络中的第一个图层，然后使用weight.data和bias.data方法访问参数。
#还可以使用替换方法normal_和fill_来重写参数值。

#定义损失函数
loss = nn.MSELoss()
#计算均方误差使用的是MSELoss类，也称为平方L2范数。默认情况下，它返回所有样本损失的平均值。

#定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
#当我们实例化一个SGD实例时，我们需要指定优化的参数 （可通过net.parameters()从我们的模型中获得）
#和优化算法所需的超参数字典。

#训练
num_epochs = 10
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)#net()函数的输出是y_hat
        trainer.zero_grad()
        l.backward()#为什么这个l不用sum？因为loss = nn.MSELoss()中默认sum过了
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch{epoch+1},loss{l:f}')

w = net[0].weight.data
print(f'w的估计误差{true_w-w.reshape(true_w.shape)}')
b = net[0].bias.data
print(f'b的估计误差{true_b-b}')