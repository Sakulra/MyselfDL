#MLP多层感知机，也叫人工神经网络
import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化参数模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

#定义激活函数relu,实际上有api，torch.nn.ReLU()
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x,a)

#定义模型
def net(x):
    x = x.reshape((-1,num_inputs))
    #@代表矩阵乘法，相当于torch.matmul(a,b)
    # H = relu(torch.matmul(x,W1)+b1)
    # return(relu(torch.matmul(H,W2)+b2))
    H = relu(x@W1+b1)
    return(relu(H@W2+b2))

#损失函数
loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
oldd2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

oldd2l.predict_ch3(net,test_iter)

d2l.plt.show()