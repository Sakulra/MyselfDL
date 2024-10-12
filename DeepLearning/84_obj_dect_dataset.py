import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

#读取数据集，读取图像和标签
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    #print('csvfname:',csv_fname)#csvfname: ..\..\dataset\banana-detection\bananas_train\label.csv
    csv_data = pd.read_csv(csv_fname)
    #set_index将Dataframe结构的数据中某一列的内容(而非列的索引名字)设置为行索引。
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        #把图片全部读到内存里
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

#创建一个自定义Dataset实例来加载香蕉检测数据集
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        #把所有数据都读进来
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        #读取第idx个样本
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        #返回数据集多长
        return len(self.features)
    
#为训练集和测试集返回两个数据加载器实例
def load_data_bananas(batch_size):
    """加载香蕉检测数据集，返回的是(训练数据集，测试数据集)这一个元组"""
    #这个返回的批量和图片分类不太一样，因为每一张图片里可能有多个物体
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
#每一个batch包含数据和标签，batch[0] 是包含图像的张量，batch[1] 是包含目标标签的张量
print(batch[0].shape, batch[1].shape)
#输出torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])
#(32个图片，RGB通道，图片高，图片宽)；(32个图片，一个物体，一个标号和四个框的坐标)

#permute(0, 2, 3, 1) 用来转换图像的维度顺序，从 (batch_size, channels, height, width) 转换为 
#(batch_size, height, width, channels)，这样可以适应常用的图像展示格式。
#除255 是将图像像素值从 [0, 255] 的范围缩放到 [0, 1]，以便可视化。
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
#画图片
axes = d2l.show_images(imgs, 2, 5, scale=2)
#在图片上画标签
#label它是一个二维矩阵但是只有一行，这一行内记录了图片标号和边框坐标
for ax, label in zip(axes, batch[1][0:10]):
    #第一列是标号不需要，只需要拿出后四列
    #如果写label[1]报错IndexError: index 1 is out of bounds for dimension 0 with size 1
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

d2l.plt.show()