import torch
from torch import nn
from d2l import torch as d2l

#AlexNet比LeNet更深更大得到了更好的精度，当变更深更大时我们有三个选项
#更多的全连接层(太贵)；更多的卷积层；将卷积层组合成块

#VGG就是将卷积层激活函数池化层组合成一个块，然后使用块连块变深，块可能不太一样。

def vgg_block(num_convs:'卷积层的个数', in_channels, out_channels):
    layers = []
    #_是num_convs里的值，有几个元组加几层，仅用来计数的。
    for _ in range(num_convs):
        #使用kernel=3x3，padding=1的卷积使输入和输出大小一样
        #之所以不用5x5的卷积核是因为经试验，在网络更深同等计算量下3x3的效果会更好
        #in_channels,和out_channels数值其实是一样的
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

#五大块，非常经典，块数基本不会变，但可以改变每块里的卷积层数
#1层卷积，64个通道；1层卷积，128通道；......
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分，和AlexNet一样
        #224x224 VGG 5次之后就变成7x7
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

#因为上面完整的VGG-11比AlexNet计算量更大，因此构建了一个通道数较小的网络，也就是简化版
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())