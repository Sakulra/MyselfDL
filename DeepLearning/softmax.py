import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#将28x28的矩阵压成784的行向量，这样仅考虑了位置关系，未考虑空间结构特征
#初始化模型参数,权重用正态分布初始化，偏置b是1x10的行向量。
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
# X = torch.normal(0,1,(2,5))
# X_prob = softmax(X)
# print(X_prob,X_prob.sum(1))

#定义模型
def net(X):
    """将输入与权重相乘再加上偏置后进行softmax化"""
    #X共有batch_size行也就是256行，每一行代表一张图片
    #x每一行乘完w后就是个行向量,刚好和b同形状。w.shape[0]是784，这里之所以能和b相加时利用了广播原理
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

#定义损失函数
def cross_entropy(y_hat,y):
    """交叉损失函数，返回预测结果与真实结果的交叉损失"""
    return -torch.log(y_hat[range(len(y_hat)),y])
#因为∑yic*log(pic),yic是符号函数(0或1)，如果样本i的真实类别等于c取1，否则取 0,所以就只剩log函数了，只需要对预测概率求log就是交叉熵了
#len(y_hat)表示有几行，（每行挑一次）
#外层加上range表示（总共有几行就挑一次）,因为len(y_hat)即y_hat的行数和y的长度也即列数一样，所以range返回从一个从零开始和y同型的矩阵，
#比如y=[1,2,3],则len(y_hat)=3,rang(len(y_hat))返回[0,1,2],就指定了y_hat的0，1，2行，y又是[1,2,3]，结合来看
#就是取出y_hat的第零行第一列，第一行第二列，第二行第三列的元素
#每行挑选哪一个元素，由y决定

#测试代码
# y = torch.tensor([0,2])
# y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# print(y_hat[[0,1],y])
#y_hat[[0, 1], y] 中的y更像是负责挑选y_hat中每行商品的"顾客"。
# print(cross_entropy(y_hat, y))

#分类精度,解释https://zhuanlan.zhihu.com/p/411852287
def accuracy(y_hat,y):
    """返回预测对的总数，并没有除以总预测数"""
    # len是查看矩阵的行数
    # y_hat.shape[1]就是列数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #y_hat每一行代表一张图片代码所认为分类的概率，这一步是为了得到每一行代码所判断的分类
        #比如[[0.1,0.3,0.6],[0.3,0.2,0.5]]-->tensor([2,2])
        y_hat = y_hat.argmax(axis=1)
        #print(type(y_hat),type(y))
    #先把y_hat换成和y一样的数据类型，然后比较y_hat和y是否在每一个位置上的值相等，使用cmp函数存储bool类型
    #y_hat和y都是tensor不用转换也行的
    cmp = y_hat.type(y.dtype) == y
    #将cmp转化为y的数据类型再求和——得到找出来预测正确的类别数
    return float(cmp.type(y.dtype).sum())

#接测试代码
# print(accuracy(y_hat,y)/len(y))

#将上面的accuracy函数进行泛化
#对于任意数据迭代器data_iter可访问的数据集,评估在任意模型net的精度
def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        net.eval()#将模型设置为评估模式,就不用计算梯度了
    metric = Accumulator(2)#列表存储正确预测数，预测总数。实际上metric=[预测正确数，总预测书]，每进行一次循环都在这两个数上累加，所以是2。
    with torch.no_grad():
        for X,y in data_iter:
            # 1、net(X)：X放在net模型中进行softmax操作
            # 2、accuracy(net(X), y)：再计算所有预算正确的样本数
             # numel()函数：返回数组中元素的个数，在此可以求得样本数
            metric.add(accuracy(net(X),y),y.numel())
            #print(metric.data)
    return metric[0]/metric[1]

#定义一个实用程序类Accumulator，用于对多个变量进行累加
class Accumulator:
    """列表包含n个变量，然后在n个变量上分别累加"""
    def __init__(self,n) -> None:#n是待累加的变量的个数
        self.data = [0.0]*n#有几个数，就把列表扩成相应大小,data 用于存储待累加的变量，如正确率、损失值，样例数等
        #如果不把data初始化为n个0就无法进行下一步的累加了，data列表是空的话，怎么让data的内容与传入的args的内容进行加法

    def add(self,*args:'list'):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
        #对于index i 从data中取出a，从args中取出b，然后相加，结果放在data 的i索引位置

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

#训练，即训练的核心部分，包括损失函数，优化函数
def train_epoch_ch3(net:'function',train_iter,loss:'function',updater:'function'):
    """训练模型一个迭代周期,返回预测正确率和预测错误率"""
    #将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()#告诉pytorch要计算梯度

    #训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)#metric=[损失函数总和，正确预测总数，总预测数]

    for X,y in train_iter:
        #计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat,y)#在这里loss就是cross_entropy()这个函数
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
            # 如果是自我实现的话，l出来是向量，我们先做求和，再求梯度,梯度清零已经包含在自我实现的优化函数里了
            #metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())#metric.add()其实是可以分情况分为上面两种，合起来的话就是这句。
    #返回训练损失和训练精度
    #metric[0]:分类正确的样本数，metric[1]:总的样本数
    return metric[0] / metric[2],metric[1] / metric[2]

#定义一个在动画中绘制数据的实用程序类Animator，它可以动态显示结果
class Animator:
    """只有一个画图区域。在动画中绘制数据,参数为xlabel;ylabel;legend;xlim:'limit of axis x'=None;ylim:'limit of axis y'=None;
    xscale='linear',yscale='linear',figsize=(3.5,2.5);fmts:'set style of lines'=('-','m--','g-','r-');nrows=1;ncols=1"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear',yscale='linear',fmts=('-','m--','g-.','r:'),
                 nrows=1, ncols=1, figsize=(3.5,2.5),) -> None:#四条线，实线，洋红虚线，绿实线，红点线,后面用了zip()所以实际用不到四条的话多的就会被舍弃
        #增量的绘制多条线
        if legend is None:
            legend = []

        d2l.use_svg_display()

        self.fig, self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        # 判断是否只有1个子图，本来初始化默认就是一个画图区域
        if ncols*nrows == 1:
            self.axes = [self.axes,]#一个子图的话也把它转化为列表，为了跟多个子图保持一致性
        # 使用lambda函数捕获参数，直接调用d2l.set_axes函数配置子图属性
        self.config_axes = lambda: d2l.set_axes(
            axes=self.axes[0], xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale, legend=legend)
        self.X,self.Y,self.fmts = None,None,fmts#X,Y是矩阵

    def add(self,x,y):#x,y存储的时本批次的值，X,Y存储的是到目前为止所有批次的值
        #像列表中添加多个数据点
        # 判断y是否包含可迭代的对象，若不是，则将其转换为一个只包含单个元素的列表
        if not hasattr(y,"__len__"):
            y = [y]#相当于y=list(y)
        n = len(y)
        # 判断参数x是否是可迭代的对象，如果不是，则将其转换为一个只包含x值的列表，并重复n次，以匹配y的长度
        if not hasattr(x,"__len__"):
            x = [x]*n
        # 如果self.X为空，创建一个包含n个空列表的列表，用于存储每个数据点的x值，下同
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        # 遍历参数x和y的元素对，将非None的值分别添加到对应的self.X和self.Y列表中的对应位置
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)#将矩阵X索引i的位置变为a，内空列表就变成一个值了
                self.Y[i].append(b)
        self.axes[0].cla()#Clear the current axes.
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)#在第一个区域进行绘图
        self.config_axes()#对第一个区域画布进行设置参数
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
        #train_epoch_ch3：训练模型，返回准确度和错误度
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)#训练一次后net()中的w和b参数已经发生改变
        #在测试数据集上评估精度
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))#y包含三个参数，有train_metrics的预测正确率，预测错误率，再加上test_acc
    train_loss,train_acc = train_metrics#这个train_loss和loss和训练中的loss函数不是一个
    # assert断言函数是对表达式布尔值的判断，要求表达式计算值必须为真。可用于自动调试。如果表达式为假，触发异常；如果表达式为真，不执行任何操作。
    assert train_loss < 0.5,train_loss
    assert train_acc <= 1 and train_acc > 0.7,train_acc
    assert test_acc <= 1 and test_acc > 0.7,test_acc

lr = 0.1
def updater(batch_size):
    """优化函数：随机梯度下降sgd"""
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
    trues = d2l.get_fashion_mnist_labels(y)#已经从索引转化为名字了
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true,pred in zip(trues,preds)]
    #print(X.shape)
    d2l.show_images(
        X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n]
    )

predict_ch3(net,test_iter)
d2l.plt.show()#显示predict_ch3(net,test_iter)的结果