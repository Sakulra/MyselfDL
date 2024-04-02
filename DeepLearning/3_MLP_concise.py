import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#简洁实现和零实现网络完全一样

#模型
#两个全连接层，第一个层是隐藏层，包含256个隐藏单元，并使用了ReLU激活函数。 第二层是输出层
net  = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10))

def init_weight(m):
    #如果是nn.Linear就对此函数权重初始化，每一层都有w和b
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

#对整个网络进行权重初始化，仅对某些层的w和b进行初始化
net.apply(init_weight)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr, batch_size = 10, 0.1, 256
#优化函数
#net.parameters()
updater = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
oldd2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.plt.show()