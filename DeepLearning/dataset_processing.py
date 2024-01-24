import torch
import torchvision
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
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../dataset", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../../dataset", train=False, transform=trans, download=True)

#数据集中包含的图片书
print(len(mnist_train), len(mnist_test))
#每张图片的大小，每张图片就相当于一个正方体横着一片一片切下来的片
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels:list) ->list[str]:
    """用于在数字标签索引及其文本名称之间进行转换。给0-9返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt','trouser','pullover','dress','coat',
                   'sandal','shirt','sneaker','bag','ankel boot']
    return [text_labels[int(i)] for i in labels]

#可视化样本
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    #
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            #图片张量
            ax.imshow(img.numpy())
        else:
            #PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes 
    
X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
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