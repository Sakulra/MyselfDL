import torch
import torchvision
import time
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt 

d2l.use_svg_display()

class Timer:
    """这个类定义了自启动计时，手动启动计时、停止，返回平均时间，返回累计时间"""
    def __init__(self) -> None:
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

#读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
#mnist_train是n行两列的矩阵，第一列数据，第二列标签
#transform参数: 指定数据预处理的方法，这里使用了transforms.ToTensor()。
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../dataset", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../../dataset", train=False, transform=trans, download=True)

#数据集中包含的图片数
print(len(mnist_train), len(mnist_test))
#每张图片的大小，每张图片就相当于一个正方体横着一片一片切下来的片
#mnist_train[0][0]表示第一张图片的数据，mnist_train[0][1]表示第一张图片的标签
#数据集中的每一张图片都以（C，H，W）的格式存储，即(通道数，高度，宽度)
print(mnist_train[0][0].shape)
#print(mnist_train[0][1])#得到一个int数字

def get_fashion_mnist_labels(labels:list) ->list[str]:
    """用于在数字标签索引及其文本名称之间进行转换。给0-9返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt','trouser','pullover','dress','coat',
                   'sandal','shirt','sneaker','bag','ankel boot']
    return [text_labels[int(i)] for i in labels]

#可视化样本，法一
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    """绘制图像列表"""
    #xscale和yscale函数的作用都是设置坐标轴的缩放类型，默认的坐标轴缩放类型为："linear"线性
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)#返回的axes是num_rows行num_cols列的矩阵
    # print(axes)
    axes = axes.flatten()
    # print(axes)
    #i就是为了对图片的数字编号转换成它的真实名字
    for i,(ax,img) in enumerate(zip(axes,imgs)):#其实省略了data这个中间变量，for i,data in enumerate(zip(axes,imgs))
                                                                           #ax,img=data
        if torch.is_tensor(img):#检查类型与python自带的isinstance(obj, type)效果一样，如果检查出obj是type类型返回true，否则返回false。
            #图片张量
            ax.imshow(img.numpy())#imshow用于显示图像。
        else:
            #PIL图片
            ax.imshow(img)
        #隐藏坐标轴
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes 
#可视化样本，法二,但是下面调用的代码需要进行一些修改
def show_images2(imgs,num_cols,titles):
    # display.set_matplotlib_formats('svg')
    d2l.use_svg_display()
    num_rows = int(len(imgs)/num_cols)
    _,figs = plt.subplots(num_cols,num_rows)
    figs = figs.flatten()
    for f,img,lbl in zip(figs,imgs,titles):#zip将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        f.imshow(img.view(28,28).numpy())#view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


images,num_labels = next(iter(data.DataLoader(mnist_train,batch_size=18)))
#num_labels是数字标号，其实就是做数据集的时候把相应的label英文名转换为数字
#print(num_labels)#输出tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5, 0, 9, 5, 5, 7, 9, 1, 0])
show_images(images.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(num_labels))
d2l.plt.show()

batch_size = 256
#读取小批量
def get_dataloader_workers() ->int:
    """使用四个进程来读取数据,其实没必要专门定义这个函数,因为在data.Dataloader()
    中可以指定工人数"""
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer = d2l.Timer()
for x,y in train_iter:
    continue
print(f'{timer.stop():2f}sec')

#整合所有组件
def load_data_fashion_minist(batch_size,resize=None):
    """加载数据集，并可选择调整数据集中每个数据的大小，最后将数据集使用dataloader()返回"""
    #这里放在列表里面是为了后面如果resize部位none，好在转换列表添加转换操作
    trans=[transforms.ToTensor()]
    #这里创建了一个数据转换列表 trans。如果 resize 不为 None，
    #则在转换列表的最前面插入一个 transforms.Resize(resize) 转换。
    #然后，通过 transforms.Compose(trans) 创建一个组合转换，这样可以按照列表中的顺序应用这些转换。
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=False, transform=trans, download=True)
    return(data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_dataloader_workers()))

train_iter,test_iter = load_data_fashion_minist(batch_size=32,resize=64)
for x,y in train_iter:#x是数据，y是标签
    print(x.shape,x.dtype,y.shape,y.dtype)
    break