import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    #平均池化
    nn.AvgPool2d(kernel_size=2, stride=2),
    #将矩阵展成一维向量，便于全连接层计算
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     #给一个输入，可以看到输入在里面形状的变化。当网络比较深的时候，不知道下一层该填多少就可以让他打印一下就知道了。
#     X = layer(X)
#     #layer.__class__.__name__是每次层的名字
#     print(layer.__class__.__name__,'output shape: \t',X.shape)

#模型训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            #如果没有给device，就把net的参数构建成一个iter，然后拿出第一个network的参数所在的device
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            #如果X是一个list就一个一个挪过去，如果是tensor那就一下就挪好了，X一般不是list。
            if isinstance(X, list):
                # BERT微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            #xavier_uniform_()初始化使输入和输出的方差相同，避免梯度消失和爆炸。
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    #将net的参数搬到gpu上
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    #Dataloader处理后就是一个列表，列表里的每一个元素包含batch_size个‘输入标签对’，所以只需要len列表长度就知道num_batches了。
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # [训练损失之和，训练正确数之和，样本数]
        metric = d2l.Accumulator(3)
        #将模型设置为训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                #损失乘以样本个数，在算train_l时又除以样本个数了
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            #乘了又除，真多此一举
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            #//取整运算，%求余运算
            #i从0开始，i+1就是num_batches,就是总数据一共被分成了多少份,无论多少份每一轮训练只添加5个记录点位。
            #对num_batches//5求余就是为了控制添加记录点位个数的，如果num_batches不对任何书取整，那只记录开始和结束的点位。
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                #每个epoch之间都被分成了五段
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        #每一轮训练完才进行测试集检验
        animator.add(epoch + 1, (None, None, test_acc))
    #最后训练完了，把一些信息输出出来
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()