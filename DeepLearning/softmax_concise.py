import torch
from torch import nn
from d2l import torch as d2l
import softmax as hand
#其实只对网络，损失函数以及优化函数调用pytorch的api其他和从零开始一样的

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化参数模型
# PyTorch不会隐式地调整输入的形状。因此，
# 在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

#softmax实现
loss = nn.CrossEntropyLoss(reduction='none')

#优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

#训练

num_epochs = 10
hand.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

