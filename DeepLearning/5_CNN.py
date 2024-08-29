#卷积后得到输出的大小:输入为h行w列，kernel时x行y列，则输出就是h-x+1行，w-y+1列
#卷积层将输入和核矩阵进行交叉相关，加上偏移后得到输出，核矩阵和偏移可学系的参数。
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算，也就是二维卷积"""
    #获得核的行和列
    h, w = K.shape
    #初始化输出的形状
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            #从X中取出和核相同大小的区域与核进行点积
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

#验证卷积核是不是和预想的一样
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

#图像中的目标边缘检测    
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

#当核覆盖的区域从0->1,还是从1->0,卷积过后要么是1要么是-1，当一样的时候就会是0，这样边缘就检测出来了
#但是只能检测垂直边缘
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)

#学习卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
#nn.Conv2()和上面自己手写的唯一区别是有输入通道和输出通道
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    #这个就相当于是model.zero_grad,model里面只有weight有梯度
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    #裸写的梯度下降
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

#查看学习10轮之后学习到的核，已经非常接近真实值了
print(conv2d.weight.data.reshape((1, 2)))