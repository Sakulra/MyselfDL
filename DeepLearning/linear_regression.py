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
def data_iter(batch_size,features,labels):#打乱数据集中的样本并以小批量方式获取数据。
    """该函数接收批量大小、特征矩阵和标签向量作为输入,生成大小为batch_size的小批量。每个小批量包含一组特征和标签"""
    num_examples=len(features)
    indices=list(range(num_examples))
    #列表中的样本顺序就被打乱了
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        #print(features[batch_indices])
        yield features[batch_indices],labels[batch_indices]

batch_size=10

for X,y in data_iter(batch_size,features,labels):
    #print(X,'\n',y)
    print('...........')
    break