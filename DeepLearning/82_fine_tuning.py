#微调是计算机视觉中最重要的部分
#除了最后一层，其他网络的结构与某一训练好的网络一模一样，只对最后一层随机初始化，而其他层的参数直接复制训练好的
#网络，这个叫做微调。然后再使用微调厚的网络用自己的数据集再稍微训练一下，就可以达到差不多的程度。
#微调后的训练，学习率以及数据迭代一般较小，而且微调的结果往往比较好。

import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用RGB通道的均值和标准差，以标准化每个通道,之所以做这个，是因为imagenet上做了。
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#最后又做了normalize是因为imagenet做了。
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

#下载imagenet预训练好的resnet18，pretrained为True就把训练好的权重参数也拿过来。
pretrained_net = torchvision.models.resnet18(pretrained=True)
#预训练的源模型实例包含许多特征层和一个输出层fc。 此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调
print(pretrained_net.fc)

#定义自己的微调网络。将预训练的resnet18的最后一层改为一个输出为2的线性层，然后再xavier初始化。
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

#微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    #如果param_group为True，则控制不同层的学习率
    if param_group:
        #params_1x将网络中除了最后一层的参数挑出来
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        #注意看大括号这里传入的参数有两个params_1x，它的学习率是默认的；net.fc.parameters()的学习率是默认的十倍。
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

#使用较小的学习率进行训练
train_fine_tuning(finetune_net, 5e-5)

# #与之对比，定义了一个相同的模型，但是将其所有模型参数初始化为随机值，因此需要大的学习率
# scratch_net = torchvision.models.resnet18()
# scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
# train_fine_tuning(scratch_net, 5e-4, param_group=False)