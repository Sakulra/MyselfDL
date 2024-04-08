import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#生成数据集y=5+1.2x-3.4x**2/2!+5.6x**3/3!
max_degree = 20 #多项式最大阶数
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 全零，x的系数，因为x多项式最大阶数是20，分配大量的空间,应该是一维数组

#从y的表达式中生成标签数据，x多项式真正的系数
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

print('true_w:',true_w)
#输入数据，一共n_train+n_test行，一列,features就是x，poly_features就是没有除以阶乘的y
features = np.random.normal(size=(n_train+n_test,1))

np.random.shuffle(features)
#power(x,y)计算x的y次方
#poly_features就是标签，每一行就是一个没有除以阶乘的y，每一行包含20个对象，分别为x的0次方到19次方
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))

for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1) #gamma(n)=(n-1)!

#np.dot(x,y)中x和y可以不是一维，torch.dot(x,y)中x，y必须是一维
labels = np.dot(poly_features,true_w)#labels为n_train+n_test行20列

print('数组labels:',labels[:2])

#加入噪声项ε它服从正态分布N（0，0.01²）
labels += np.random.normal(scale=0.1,size=labels.shape)
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

#features[:2], poly_features[:2, :], labels[:2]

def load_array(data_arrays, batch_size, is_train=True):
    """加载自建数据集"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

#模型评估
def evaluate_loss(net,data_iter,loss):
    metric = d2l.Accumulator(2) #损失总和，样本数量
    for X,y in data_iter:
        out = net(X)
        #y = y.reshape(out)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0] / metric[1]

#训练函数
def train(train_features,test_features,train_labels,test_labels,num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    #train_iter实际上是一个迭代器，类似于列表之类的，每个对象包含batch_size个元素，每处理完一个对象就会迭代下一个对象。
    train_iter = load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)
    updater = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[1,num_epochs],ylim=[1e-3,1e2],legend=['train','test'])
    for epoch in range(num_epochs):
        oldd2l.train_epoch_ch3(net,train_iter,loss,updater)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1,(evaluate_loss(net,train_iter,loss),
                                    evaluate_loss(net,test_iter,loss)))
        print('weight:',net[0].weight.data.numpy())

# #三阶多项式，从多项式特征中选择前四个维度，1，x,x**2/2!,x**3/3!
# train(poly_features[:n_train,:4],poly_features[n_train:,:4],
#       labels[:n_train],labels[n_train:])

# #线性函数拟合（欠拟合）
# #从多项式特征中选择前两个维度，1，x
# train(poly_features[:n_train,:2],poly_features[n_train:,:2],
#       labels[:n_train],labels[n_train:])

# #高阶多项式函数拟合，从多项式特征中选取所有维度
# train(poly_features[:n_train,:],poly_features[n_train:,:],
#       labels[:n_train],labels[n_train:],num_epochs=1500)

d2l.plt.show()