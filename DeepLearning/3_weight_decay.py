#高维线性回归
import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#生成数据y=0.05+∑0.01xi+ε，ε∈N（0，0.01**2）
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs,1))*0.01, 0.05

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

train_data = synthetic_data(true_w,true_b,n_train)
train_iter = load_array(train_data,batch_size)
test_data = synthetic_data(true_w,true_b,batch_size)
test_iter = load_array(test_data,batch_size,is_train=False)

#初始化参数模型
def init_params():
    w = torch.normal(0, 1, size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

#L2范数
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#优化函数
def updater(batch_size,w,b,lr):
    """优化函数：随机梯度下降sgd"""
    return d2l.sgd([w,b],lr,batch_size)

#训练代码
def train(lambd:"λ",updater):
    w, b = init_params()
    #net = lambda X: d2l.linreg(X, w, b)
    net = lambda X: torch.matmul(X,w) + b
    loss = d2l.squared_loss
    num_epoch, lr = 100, 0.03
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[5,num_epoch],legend=['train','test'])
    for epoch in range(num_epoch):
        for X, y in train_iter:
            #增加L2范数惩罚项
            #广播机制使l2_penalty变为batch_size的向量
            l = loss(net(X),y) + lambd*l2_penalty(w)
            l.sum().backward()
            #d2l.sgd([w,b],lr,batch_size)
            updater(batch_size,w,b,lr)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net,train_iter,loss),
                                    d2l.evaluate_loss(net,test_iter,loss)))
        print('w的L2范数是：', torch.norm(w).item())

#忽略正则化直接训练，必过拟合
train(lambd=0,updater=updater)
#使用权重衰减
train(lambd=3,updater=updater)
d2l.plt.show()

#简洁实现####################################################################################################
def train_concise(weightdecay:bool):#weightdecay是一个bool值，为1的话就权重衰减
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    #初始化参数w,b
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    #优化函数，惩罚项既可以写在目标函数里面，也可以写在训练算法里面，每一次在更新前把当前的w诚意衰减因子weight_decay
    #因为有两个参数需要优化但是优化要求不一样所以有两个字典，对net[0]的weight启用参数衰减，而bias使用默认的
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': weightdecay},
        {"params":net[0].bias}], lr=lr)
    trainer = torch.optim.SGD([
        {"params":net[0]}
    ])
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

#train_concise(0)