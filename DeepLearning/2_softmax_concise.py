import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l
#其实只对网络，损失函数以及优化函数调用pytorch的api其他和从零开始一样的

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化参数模型
# PyTorch不会隐式地调整输入的形状。
# 因此，在线性层前定义展平层（flatten），来调整网络输入的形状
#一共有两个层，展开层和线性层
#实际输入的数据是(256，28，28)将其展平后变为(256,28*28=784),后边再输入nn.Linear()
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    #对某些特定的子模块做一些针对性的处理
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)#方差为0.01

net.apply(init_weights)#net网络的参数用init_weights初始化

#softmax实现
loss = nn.CrossEntropyLoss(reduction='none')

#优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

#训练

num_epochs = 10
oldd2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()