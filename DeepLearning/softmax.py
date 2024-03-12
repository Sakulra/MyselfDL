import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#将28x28的矩阵压成184的行向量，这样仅考虑了位置关系，未考虑空间结构特征
#初始化模型参数,权重用正态分布初始化，偏置b是1x10的行向量，不知道b怎么加入计算的？
#w每个类别的权重占一列，每一列包含784个权重参数，共10类，所以是784x10。
num_inputs = 784
num_outputs = 10
w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

#定义softmax操作
#实现softmax分三步
#1.对每个项求幂。
#2.对每一行求和，得到每个样本的规范化常数。
#3.将每一行除以其规范化常数，确保结果的和为1.
def softmax(X):
    """对输入的矩阵进行softmax化,使每一行之和为1,输入其实就是一张图片也就是一行"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition

#测试是否得到预期结果
X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
print(X_prob,X_prob.sum(1))

#定义模型
def net(X):
    """将输入与权重相乘再加上偏置后进行softmax化"""
    #x如果是一张图片的话正好，x乘完w后就是个行向量,刚好和b同形状
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

#定义损失函数
def cross_entropy(y_hat,y):
    """交叉损失函数，返回预测结果与真实结果的交叉损失"""
    return -torch.log(y_hat[range(len(y_hat)),y])
#len(y_hat)表示有几行，（每行挑一次）
#外层加上range表示（总共有几行就挑一次）,因为len(y_hat)即y_hat的行数和y的长度也即列数一样，所以range返回从一个从零开始和y同型的矩阵，
#比如y=[1,2,3],则len(y_hat)=3,rang(len(y_hat))返回[0,1,2],就指定了y_hat的0，1，2行，y又是[1,2,3]，结合来看
#就是取出y_hat的第零行第一列，第一行第二列，第二行第三列的元素
#每行挑选哪一个元素，由y决定

#测试代码
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])
#y_hat[[0, 1], y] 中的y更像是负责挑选y_hat中每行商品的"顾客"。
print(cross_entropy(y_hat, y))

#分类精度,解释https://zhuanlan.zhihu.com/p/411852287
def accuracy(y_hat,y):
    # len是查看矩阵的行数
    # y_hat.shape[1]就是去列数，y_hat.
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)#y_hat每一行代表一张图片代码所认为分类的概率，这一步是为了得到每一行代码所判断的分类
                                    #比如[[0.1,0.3,0.6],[0.3,0.2,0.5]]-->tensor([2,2])
    cmp = y_hat.type(y.dtype) == y#先把y_hat换成和y一样的数据类型，然后比较y_hat和y是否在每一个位置上的值相等
    return float(cmp.type(y.dtype).sum())

#接测试代码
print(accuracy(y_hat,y)/len(y))

#对于任意数据迭代器data_iter可访问的数据集,评估在任意模型net的精度
def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        net.eval()#将模型设置为评估模式,就不用计算梯度了
    metric = Accumulator(2)#存储正确预测数，预测总数
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

#定义一个实用程序类Accumulator，用于对多个变量进行累加
class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n) -> None:#n是待累加的变量的个数
        self.data = [0.0]*n#有几个数，就把列表扩成相应大小,data 用于存储待累加的变量，如正确率、损失值，样例数等
        #如果不把data初始化为n个0就无法进行下一步的累加了，data列表是空的话，怎么让data的内容与传入的args的内容进行加法

    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
        #对于index i 从data中取出a，从args中取出b，然后相加，结果放在data 的i索引位置

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

#训练
def train_epoch_ch3(net,train_iter,loss,updater):
    """训练模型一个迭代周期"""
    #将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()#告诉pytorch要计算梯度

    #训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)

    for X,y in train_iter:
        #计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            #使用pytorch内置的优化器和损失函数
            updater.zero_grad()#先把梯度设置为零
            l.mean().backward()#计算梯度
            updater.step()#自更新
            #metric.add(float(l) * len(y), accuracy(y_hat, y),y.size().numel())
        else:
            #使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            # 如果是自我实现的话，l出来就是向量，我们先做求和，再求梯度
            #metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())#metric.add()其实是可以分情况分为上面两种，合起来的话就是这句。
    #返回训练损失和训练精度
    #metric[0]:分类正确的样本数，metric[1]:总的样本数
    return metric[0] / metric[2],metric[1] / metric[2]

#定义一个在动画中绘制数据的实用程序类Animator，它可以动态显示结果
class Animator:
    """在动画中绘制数据,参数为xlabel;ylabel;legend;xlim:'limit of axis x'=None;ylim:'limit of axis y'=None;
    xscale='linear',yscale='linear',figsize=(3.5,2.5);fmts:'set style of lines'=('-','m--','g-','r-');nrows=1;ncols=1"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear',yscale='linear',fmts=('-','m--','g-.','r:'),
                 nrows=1, ncols=1, figsize=(3.5,2.5),) -> None:
        #增量的绘制多条线
        if legend is None:
            legend = []

        d2l.use_svg_display()

        self.fig, self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if ncols*nrows == 1:
            self.axes = [self.axes,]#
        #使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X,self.Y,self.fmts = None,None,fmts

    def add(self,x,y):
        #像列表中添加多个数据点
        if not hasattr(y,"__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x,"__len__"):
            x = [x]*n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()#Clear the current axes.
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)

        d2l.plt.draw()
        d2l.plt.pause(0.01)

        display.clear_output(wait=True)

#训练函数它会在train_iter访问到的训练数据集上训练一个模型net。 
#该训练函数将会运行多个迭代周期（由num_epochs指定）。 在每个迭代周期结束时，
#利用test_iter访问到的测试数据集对模型进行评估。 我们将利用Animator类来可视化训练进度。
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1,num_epochs], ylim=[0.3,0.9],
                        legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc = train_metrics
    assert train_loss < 0.5,train_loss
    assert train_acc <= 1 and train_acc > 0.7,train_acc
    assert test_acc <= 1 and test_acc > 0.7,test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([w,b],lr,batch_size)
num_epochs = 10
#########################错在这最后一步了，我重看四五遍，硬是没发现啊，裂开.....................##########
#train_ch3后面不是等号，它是个函数调用啊，所以每次执行代码的时候会迅速结束，因为根本没调用神经网路进行计算
#train_ch3 = (net,train_iter,test_iter,cross_entropy,num_epochs,updater)
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)

#预测
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true,pred in zip(trues,preds)]
    d2l.show_images(
        X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n]
    )

predict_ch3(net,test_iter)
d2l.plt.show()#显示predict_ch3(net,test_iter)的结果