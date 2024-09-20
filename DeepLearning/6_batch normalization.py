#批量归一化，固定小批量中的均值和方差，然后学习出适合的偏移和缩放。可以加快收敛速度，但不改变模型精度。
#xi+1=γ(xi-μ)/σ +β效果之所以好，可能就是通过在每一个小批量里加入噪音来控制模型复杂度。
#因为对于每一个批次,它计算出来得均值和方差是不一样的,所以对于xi+1它引入了随机偏移和随机缩放,因此无需跟丢弃法混合使用。
#因为每层的输入都服从相近的分布，学习率就可以调的大一点，无需担心上层梯度太大或者底层梯度太小。

import torch
from torch import nn
from d2l import torch as d2l

#gamma, beta是可以学习的两个参数，moving_mean, moving_var全局的均值和方差，在做推理的时候用。
#momentum用来更新全局均值方差，eps为了避免除0的东西，它俩都是固定值，不要改！
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    #如果不是训练模式，也就是是推理模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        #X的维度必须在2、4里面，不在就报个错
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            #按行求均值，也就是每一列求出一个均值，行变列不变，最后变成1xn的行向量。
            mean = X.mean(dim=0)
            #方差也是一个行向量
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。得到1xnx1x1
            #因为(批量大小，通道数，高，宽)，要计算通道维的均值和方差，就要用到批量大小、高、宽，故是dim=(0,2,3)
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

#定义batchnorm层
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层。它只有2和4，其他情况目前不考虑。
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0，会被维护，且是学习出来的。
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1，会被维护，但不是学习出来的。
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在显存上，将moving_mean和moving_var，复制到X所在显存上
        # moving_mean，moving_var它没有放在nn.Parameter中，在使用net.to(device)的时候不会自动挪过去。
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
    
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))

# #使用框架的batchnorm
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#     nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
#     nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
#     nn.Linear(84, 10))

# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())