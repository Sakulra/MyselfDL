#包括数据预处理、模型设计和超参数选择
import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from d2l import oldtorch as oldd2l

#建立字典DATA_HUB,将数据集名称的字符串映射到数据集相关的二元组上,二元组包含数据集的url和验证文件完整性的sha-1密钥.
#字典形式{'数据集名'：(url,sha1_hsah)}，所有类似的数据集都托管在地址为DATA_URL的站点上.
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
#训练集和数据集
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

#下载数据集
def dowload(name,cache_dir=os.path.join('../../dataset','data')):
    """下载字典DATA_HUB重的文件，并返回本地文件名"""
    assert name in DATA_HUB,f"{name}不存在于{DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)#在当前目录创建data文件夹
    fname = os.path.join(cache_dir,url.split('/')[-1])#创建csv文件名，便于后面好写入数据
    #print(cache_dir)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname#命中缓存
    print(f'正在从{url}下载{fname}')
    r = requests.get(url,stream=True,verify=True)
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname

def downloas_extract(name,folder=None):
    """下载并解压zip/tar文件"""
    fname = dowload(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splittext(fname)
    if ext == 'zip':
        fp = zipfile.ZipFile(fname,'r')
    elif ext in (',tar','.gz'):
        fp = tarfile.open(fname,'r')
    else:
        assert False
    fp.extractall(base_dir)
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        dowload(name)

#读取数据集，训练数据集包括1460个样本，每个样本80个特征和1个标签，测试数据集包含1459个样本，每个样本80个特征
train_data = pd.read_csv(dowload('kaggle_house_train'))
test_data = pd.read_csv(dowload('kaggle_house_test'))

# print(train_data.shape)
# print(test_data.shape)
#输出前四个和最后两个特征，以及相应标签（房价）
#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#剔除数据中的第一列(id编号)
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#数据预处理：将缺失的值替换为相应特征的平均值，并且将所有特征缩放到正态分布N(0,1)。缩放方法在线代中有(x-u)/σ

#因为all_features中的对象有字符有数字，还有空，所以它不是矩阵,它是DataFrame
#将特征是数字的列滤出,针对非数字列进行处理，
#all_features.dtypes是获得每一列数据的类型进行返回,然后all_features.dtypes != ‘object’，判断各列数据是否不为object对象。不为则返回True
#object类是所有类的基类，其他类都继承自object类
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(all_features.dtypes)
#对所有特征的值进行缩放，即方便优化又平等对待每个特征
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#用独热编码代替离散值比如'MSZoning'包含值“RL”和“Rm”，将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1，pandas可以自动实现这一点。
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
#all_features = pd.get_dummies(all_features, dummy_na=True)
#某些ide独热编码时会将数据处理成boolean类型，需要指定处理类型为数值型，如int。
all_features =pd.get_dummies(all_features, dummy_na=True, dtype=int)
print(all_features.shape)
#从pandas格式中提取numpy格式，并将其转为张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

#训练
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
#评价误差函数
def log_rmse(net,features,labels):
    #为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features),1,float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()#取出单元素张量的元素数值并返回该数值

def train(net, train_features, train_labels, test_features, test_labels,num_epochs,learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features,train_labels),batch_size=batch_size)
    #使用adam优化算法,adam的优点在于对初始学习率不敏感
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls, test_ls

#K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx,:],y[idx]
        if j==i:
            X_valid, y_valid = X_part, y_part
        elif x_train is None:
            x_train, y_train = X_part, y_part
        else:
            x_train = torch.cat([x_train, X_part],0)
            y_train = torch.cat([y_train, y_part],0)
    return x_train, y_train, X_valid, y_valid

#在折交叉验证中训练k次后，返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()