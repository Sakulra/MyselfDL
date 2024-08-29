import torch
from torch import nn
from torch.nn import functional as F

#加载和保存张量..............................................................................
x = torch.arange(4)
torch.save(x, 'x-file')

#将存储在文件中的数据读回内存
x2 = torch.load('x-file')
print(x2)

#存储张量列表，然后再读回内存
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

#写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

#加载和保存模型..............................................................................
#pytorch不能把整个网络结构存储下来，而tensorflow是可以的，所以pytorch的存储只有权重，没有计算量
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

#将模型的参数存储在一个叫做“mlp.params”的文件中
#把MLP中的参数名字和参数值的映射存储为字典
torch.save(net.state_dict(), 'mlp.params')

#当要加载参数的时候必须把网络给复现了(也就是复制过来)
#为了恢复模型，实例化了原始多层感知机模型的一个备份。这里不需要随机初始化模型参数，而是直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

#由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同
Y_clone = clone(X)
print(Y_clone == Y)