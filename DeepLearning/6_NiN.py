import torch
from torch import nn
from d2l import torch as d2l

#通过对LeNet、AlexNet、VGG参数的计算，发现在全连接层的第一层占据了参数的大头，而且全连接层过宽的话还容易过拟合。
#NiN块就是最后不使用全连接层，而改为1x1的卷积层代替全连接层，本质上还是全连接层。
#NiN主要由AlexNet来
#NiN架构：无全连接层、交替使用NiN块和步幅为2的最大池化层，逐步减小高宽和增大通道数、最后使用全局平均池化层得到输出，输入通道书是类别数。

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10，所以最后输出通道为10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    #全局平均池化，输出的高宽都变成1，也就是在每一个通道里面将其平均成一个数，这个数就是它的预测类别。
    #它替换的是AleNet的最后一个Dense层
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    #将批量大小保持住，其他全部拉成一个向量
    nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())