import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#将28x28的矩阵压成184的行向量，这样仅考虑了位置关系，未考虑空间结构特征
#初始化模型参数,权重用正态分布初始化，偏置b是1x10的行向量，不知道b怎么加入计算的？
#w每个类别的权重占一列，每一列包含784个权重参数，共10类，所以是784x10。
num_input = 784
num_output = 10

w = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
b = torch.zeros(num_output,requires_grad=True)

#定义softmax操作
#实现softmax分三步
#1.对每个项求幂。
#2.对每一行求和，得到每个样本的规范化常数。
#3.将每一行除以其规范化常数，确保结果的和为1.
def softmax(X:"Matrix"):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition

#测试是否得到预期结果
# X = torch.normal(0,1,(2,5))
# X_prob = softmax(X)
# print(f'{X_prob},{X_prob.sum(1)}')

#定义模型
def net(X):
    #x如果是一张图片的话正好，x乘完w后就是个行向量,刚好和b同形状
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

#定义损失函数
def cross_entropy(y_hat:'forecast category',y:'true category') ->'loss':
    return -torch.log(y_hat[range(len(y_hat)),y])
#len(y_hat)表示有几行，（每行挑一次）
#外层加上range表示（总共有几行就挑一次）,因为len(y_hat)即y_hat的行数和y的长度也即列数一样，所以range返回从一个从零开始和y同型的矩阵，
#比如y=[1,2,3],则len(y_hat)=3,rang(len(y_hat))返回[0,1,2],就指定了y_hat的0，1，2行，y又是[1,2,3]，结合来看
#就是取出y_hat的第零行第一列，第一行第二列，第二行第三列的元素
#每行挑选哪一个元素，由y决定

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])
#y_hat[[0, 1], y] 中的y更像是负责挑选y_hat中每行商品的"顾客"。
print(cross_entropy(y_hat, y))

#分类精度
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())