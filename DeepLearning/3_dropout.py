import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#以dropout的概率丢弃节点
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    #丢弃所有元素
    if dropout == 1:
        return torch.zeros_like(X)
    #保留所有节点
    if dropout == 0:
        return X
    #用于生成一个与张量X同行的掩码mask，掩码中的元素值为0或1，dropout是介于0，1之间的概率值，用于确定掩码中0的比例。
    #对与mask中的每个元素，如果生成的随机数大于dropout，则将该元素置1.否则置0，最后将生成的mask转化为浮点型
    mask = (torch.rand(size=X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

#test
# X = torch.arange(16,dtype = torch.float32).reshape((2,8))
# print(X)
# print(dropout_layer(X,0.))
# print(dropout_layer(X,0.5))
# print(dropout_layer(X,1.0))

#定义模型参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_train = True) -> None:
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_train
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        #只有在训练模式才使用dropout
        if self.training == True:
            #在第一个全连接层忠厚添加dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            #在第二个全连接层后添加dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

#训练和测试
num_epochs,lr,batch_sze = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, tesr_iter = d2l.load_data_fashion_mnist(batch_sze)
optimizer = torch.optim.SGD(net.parameters(),lr=lr)
oldd2l.train_ch3(net,train_iter,tesr_iter,loss,num_epochs,optimizer)

#简洁实现
#在训练时，Dropout层将根据指定的暂退概率随机丢弃上一层的输出（相当于下一层的输入）。 在测试时，Dropout层仅传递数据。
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    #在第一个全连接层后加dropout
                    nn.Dropout(dropout1),
                    nn.Linear(256,256),
                    nn.ReLU(),
                    #在第二个全连接层后加dropout
                    nn.Dropout(dropout2),
                    nn.Linear(256,10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight,std = 0.01)

net.apply(init_weight)

trainer = torch.optim.SGD(net.parameters(),lr=lr)
oldd2l.train_ch3(net,train_iter,tesr_iter,loss,num_epochs,optimizer)