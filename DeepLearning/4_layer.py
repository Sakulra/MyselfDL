import torch
from torch import nn
from torch.nn import functional as F

#自定义神经网络

#自定义块............................................................................
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
    
#自定义顺序快（只需要定义两个关键函数）................................................................................
#1.一种将块逐个追加到列表中的函数；
#2.一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”
#注：__init__函数将每个模块逐个添加到有序字典_modules中；每个Module都有一个_modules属性，_module的类型是OrderedDict。
#   使用_modules而不自建列表的优点：在模块的参数初始化过程中，系统知道在_modules字典中查找需要初始化参数的子块
class MySequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for idx, module in enumerate(args):# module是Module子类的一个实例。我们把它保存在'Module'类的成员变量_modules中。
            self._modules[str(idx)] = module
            
    def forward(self, X):
        for block in self._modules.values():#OrderedDict保证了按照成员添加的顺序遍历它们
            X = block(X)
        return X
#自定义顺序快MySequential就和nn.Sequential()功能一模一样了。
net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

#在前向传播函数中执行代码(当自己想要的架构不是简单的顺序架构时Sequential类就无法满足了)....................................
#比如，在前向传播函数中执行Python的控制流；或者执行任意的数学运算，而不是简单地依赖预定义的神经网络层
class FixedHiddenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #rand_weight仅仅是一个随机权重，不会参加训练
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self,X):
        X = self.linear(X)#没错，就是我被共享了
        #mm矩阵相乘，mv矩阵向量相乘，mul矩阵点乘，matmul全能
        X = F.relu(torch.mm(X,self.rand_weight) + 1)
        #复用全连接层。这相当于两个全连接层共享参数，和上面那个self.linear(X)
        X = self.linear(X)
        #控制流
        while X.abs().sum() > 1:
            X /= 2
        return X
net = FixedHiddenMLP()

#混合搭配各种组合块，
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
###############################################################################################################
#参数管理
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand((2,4))

#访问参数状态
print(net[2].state_dict())
#访问偏置,及偏置里的数据
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
#访问梯度
print(net[2].weighr.grad)
#一次访问所有网络的参数
print(*[(name,param.shape) for name, param in net[0].named_parameters()])
print(*[(name,param.shape) for name, param in net.named_parameters()])
#访问最后一层的偏移
print(
    net.state_dict()['2.bias'].data
)


#从嵌套块收集参数...................................................................
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        #使用add_module和使用sequential的区别就是，可以往里面传字符，自己命名每一个网络的名字，而=而不是默认的0，1，2
        #比如下面的代码表示模块的名字为，block1，block2，...，block4。
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))#输出网络结构
#因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们
rgnet[0][1][0].bias.data#访问第一个主要的块中、第二个子块的第一层的偏置项


#内置初始化...............................................................................................
#默认初始化，将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0
def init_normal(m):#m就是一个module，每次传入一个
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)#在最后面是‘_’意思就是，它是一个替换函数，它不会返回一个值，而是直接替换掉传入的m.weight
        nn.init.zeros_(m.bias)
net.apply(init_normal)#对net里的所有module调用init_normal这个函数
print(net[0].weight.data[0], net[0].bias.data[0])

#将所有参数初始化为给定的常数，比如初始化为1
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)#和上面唯一的区别，将权重初始化为全1
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

#对某些块应用不同的初始化方法
#使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


#自定义初始化.....................................................................................
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5#它乘以它是不是绝对值大于五，若不是就变成0了，是的话保持

net.apply(my_init)
print(net[0].weight[:2])

#始终可以直接设置参数
net[0].weight.data[:] += 1#所有元素+1
net[0].weight.data[0, 0] = 42#第一个元素等于42
print(net[0].weight.data[0])


#参数绑定.....................................................................................
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
#net[2]和net[4]的参数是一模一样的，用的是同一个，凡是用到shared的它们用的都是用一个参数对象。
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])


###自定义层，上面的是自定义网络,但是本质没有区别，形式上也没区别##########################################################

#构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
#output：tensor([-2., -1.,  0.,  1.,  2.])

#将层作为组件合并到构建更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())

#带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):#in_units输入大小，units输出大小
        super().__init__()
        #把想初始化的参数放进nn.Parameter()中就是为了把梯度加上，并给一个合适的名字，便于后来访问参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
linear = MyLinear(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))