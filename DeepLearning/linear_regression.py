import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

#人工生成数据集
def synthetic_data(w,b,num_examples): #num_examples是样本数
    """生成y=Xw+b+噪声"""
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w) + b
    y+=torch.normal(0,0.1,y.shape)
    return x,y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=4.2

features,labels=synthetic_data(true_w,true_b,1000)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), 
                labels.detach().numpy(), 1)#detach是因为在pytorch得一些版本中需要从梯度计算图中detach出来才能进行numpy转换
#print(len(features),features.shape)#输出：1000 torch.Size([1000, 2])
plt.show()

#读取数据集
def data_iter(batch_size,features,labels) ->'return batch_size number of features and labels':
    #打乱数据集中的样本并以小批量方式获取数据。
    """该函数接收批量大小、特征矩阵和标签向量作为输入,生成大小为batch_size的小批量。每个小批量包含一组特征和标签"""
    num_examples=len(features)
    indices=list(range(num_examples))
    #列表中的样本顺序就被打乱了
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        # print(batch_indices)
        yield features[batch_indices],labels[batch_indices]

batch_size=10

for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
#正常情况下以上数据的获取不是通过代码生成的，以下部分才是代码真正开始
#初始化参数模型,这两个值是要不断更新来拟合数据的
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

#定义模型，将模型的输入和参数同模型的输出关联起来
def linreg(X:'tensor_matrix',w:'Tensor',b:'constant'):
    return torch.matmul(X,w) + b

#定义损失函数L2范数,也可用平方损失函数，区别不大
def squared_loss(y_hat,y) ->'Tensor':
    """得到Y_hat和y的平方损失的tensor"""
    return (y_hat-y)**2/2#如果y的形状不太清楚就y.reshape(y_hat.shape)

#定义优化算法，也就是对w,b进行更新，使用小批量随机梯度下降
def sgd(params:'need_update_object',lr:'learning_rate' = 0.03,batch_size:'int' = 20):
    """使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度"""
    with torch.no_grad():#告诉torch不需要计算梯度
        for param in params:   #在本例中有w,b都需要更新因此params是个列表，包含了w和b
            param -= lr*param.grad/batch_size
            param.grad.zero_()

#开始训练
#初始化参数
#重复以下训练，直至完成
    #计算梯度g
    #更新参数(w,b)=(w,b)-lr*g
lr = 0.03
num_epoch = 10
#为了解耦合,后期好换成其他神经网络或其他损失函数，而不必改主函数
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y) #X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()#backward一般对标量进行计算梯度，很少对向量计算梯度，因为对一维向量求导就会变成一个矩阵，越求导越复杂
        sgd([w,b],lr,batch_size)
        with torch.no_grad():
            train_l = loss(net(features,w,b),labels)#X和y的整体损失
            print(f'epoch {epoch+1},loss{train_l.mean():f}')

#查看训练出来的w_hat和b_hat与真实的w和b的差距
print(f'w的估计误差{true_w-w.reshape(true_w.shape)}')
print(f'b的估计误差{true_b-b}')

###########################################################################################
