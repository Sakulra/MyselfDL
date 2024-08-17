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

#建立字典DATA_HUB,将数据集名称的字符串映射到数据集相关的二元组上,二元组包含数据集的url和验证文件完整性的sha-1密钥.
#字典形式{'数据集名'：(url,sha1_hsah)}，所有类似的数据集都托管在地址为DATA_URL的站点上.
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
#训练集和数据集,字典的键是文件名，值是一个元组(网址，哈希值)
#('http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
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
    print(DATA_HUB[name])
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)#在当前目录创建data文件夹
    fname = os.path.join(cache_dir,url.split('/')[-1])#url.split('/')[-1]就是csv文件名，在加上要存的地址名。创建csv文件名，便于后面好写入数据
    #print(cache_dir)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)#读取1048576字节数据
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

print('train_data.shape',train_data.shape)
# print(test_data.shape)
#输出前四个和最后两个特征，以及相应标签（房价）
#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#将测试集和数据集连接成一个整体，并剔除训练数据中的第一列(id编号)。
#根据左闭右开。由于 -1 训练数据的最后一行也被剔除掉了
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#数据预处理：将缺失的值替换为相应特征的平均值，并且将所有特征缩放到正态分布N(0,1)。缩放方法在线代中有(x-u)/σ

#因为all_features中的对象有字符有数字，还有空，所以它不是矩阵,它是DataFrame
#object类是所有类的基类，其他类都继承自object类
#将特征是数字的列排除掉,针对非数字列进行处理。
#all_features.dtypes是获得每一列数据的类型进行返回,然后all_features.dtypes != ‘object’，判断各列数据是否不为object对象。不为则返回True
#对于DataFrame，dtypes返回一个Series，其中包含DataFrame中每列的数据类型。然后调用index刚好获得每一列的名字,index不是数字就是名字。
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index#numeric_features是Series，Series只有行索引没有列索引

#print(all_features.dtypes)

#对所有特征的值进行缩放，全部进行正态分布化N(0,1)，(x-u)/σ，即方便优化又平等对待每个特征
#这个x代表的是Series里的每一个数据，而比如x.mean()等操作是针对Series进行操作的。
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失也即为0，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#用独热编码代替离散值比如'MSZoning'列包含值“RL”和“Rm”，将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
# “Dummy_na=True”将“NA”（缺失值）视为有效的特征值，并为其创建指示符特征
#某些ide独热编码时会将数据处理成boolean类型，需要指定处理类型为数值型，如int。
#all_features = pd.get_dummies(all_features, dummy_na=True)
all_features =pd.get_dummies(all_features, dummy_na=True, dtype=int)

#print('all_features.shape',all_features.shape)

#文件是pandas读取的，通过value属性可从pandas格式中提取numpy格式，并将其转为张量。
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

# print('train_features.shape',train_features.shape)
# print('test_features.shape',test_features.shape)

train_labels = torch.tensor(#test_data没有SalePrice这一列
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

#训练
loss = nn.MSELoss()
in_features = train_features.shape[1]#输入特征个数，也就是输入神经元个数

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))#因为输出就是房价，因此输出就是一个神经元
    return net
#评价误差函数，
def log_rmse(net,features,labels):
    #对于房价，为了在取对数时进一步稳定该值，将小于1的值设置为1，clipped：修剪的。
    clipped_preds = torch.clamp(net(features),1,float('inf'))#把模型输出的值限制在1到inf之间，infinity是无穷大
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()#取出单元素张量的元素数值并返回该数值

def train(net, train_features, train_labels, test_features, test_labels, num_epochs,
          learning_rate, weight_decay, batch_size):
    """返回训练集损失train_ls和测试集损失test_ls"""
    train_ls, test_ls = [], []#用来存储训练集和测试集的训练损失函数值
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

#K折交叉验证。在k折交叉验证中，每次模型的训练都是独立于上一次的。
#K折交叉验证的步骤如下：
# 1.将原始数据集分成K个子集。
# 2.对于每个子集i，将其作为验证集，其余的K-1个子集作为训练集。
# 3.训练模型，并在验证集上进行评估，记录评估指标（如准确率、精确度、召回率等）。
# 4.重复步骤2和3，直到每个子集都充当一次验证集，从而产生了K个评估指标。
# 5.对K个评估指标取平均值作为最终的模型性能评估指标。
def get_k_fold_data(k, i, X, y):
    """获得k折交叉验证的训练集和测试集，k折，i测试集，X总数据，y总标签"""
    assert k > 1
    fold_size = X.shape[0] // k #每一折的大小为样本数除以k
    x_train, y_train = None, None#k折 后 总的训练数据。易错：必须是None而不能是空列表等其他的。
    for j in range(k):#每一折，j是从0开始的
        #python内置函数slice(start, stop, step)用于生成一个切片对象
        idx = slice(j * fold_size, (j+1) * fold_size)#每一折的切片索引间隔，也就是每一折从几到几。
        X_part, y_part = X[idx,:],y[idx]#把每一折对应的数据取出来，对应好。
        if j==i:#i表示第几折把它作为验证集
            X_valid, y_valid = X_part, y_part
        #接下来对训练集的分类处理容易出错，没想到。
        # 经过我的测试，如果x_train是空的，无法直接使用torch.cat()进行拼接，所以必须将是第一次看到X_train单独列出来处理。
        elif x_train is None:
            x_train, y_train = X_part, y_part
        else:#x_train不是空的之后可直接用torch.cat与原先的合并
            x_train = torch.cat([x_train, X_part],0)
            y_train = torch.cat([y_train, y_part],0)
    return x_train, y_train, X_valid, y_valid#返回训练集和验证集

#减少循环版的获取训练集和测试集
# def get_k_fold_data(k, i, X, y):
#     assert k > 1
#     fold_size = X.shape[0] // k
#     valid_start = i * fold_size 
#     valid_end = (i + 1) * fold_size 

#     #把训练集分割出来
#     X_head, y_head = X[:valid_start, :], y[:valid_start]
#     X_tail, y_tail = X[valid_end:, :], y[valid_end:]
#     #这个判断想不出来，没有原版好想和方便。
#     if i == 0:
#         valid_start = None      
#         X_head, y_head = torch.tensor([]), torch.tensor([])
#     elif i == k - 1:
#         valid_end = None        
#         X_tail, y_tail = torch.tensor([]), torch.tensor([])
        
#     #把测试集分割出来
#     valid_idx = slice(valid_start, valid_end)
#     X_valid, y_valid = X[valid_idx, :], y[valid_idx]
#     #把训练集连接起来
#     X_train = torch.cat([X_head, X_tail], 0)
#     y_train = torch.cat([y_head, y_tail], 0)    
#     return X_train, y_train, X_valid, y_valid

#在k折交叉验证中训练k次后，将每次训练和验证误差做平均值，然后返回
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0#训练和测试的均值
    for i in range(k):#第i折是测试集，一共k折
        data = get_k_fold_data(k, i, X_train, y_train)#获得第i折的测试集，和其它折的训练集
        #每一次k折都当成一次独立验证；如果使用同一个net，因为每一折都调用了train()且都对net参数进行更新，
        #那第i折的验证会受到j<i折验证的影响，使得验证相比实际理想化.
        net = get_net()
        # * 是解码变成前面返回的四个数据，
        # *data解开后x_train, y_train, X_valid, y_valid刚好对应train()函数的train_features,train_labels,test_features,test_labels
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        #每一折都往train_ls里添加，所以求和的时候每次取train_ls[-1]
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:#为什么是等于0的时候画图，等于0的时候不是还没有数据吗？怎么画出来的图？
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

#测试集验证
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay,batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1),[train_ls],xlabel='epoch',
             ylabel='log rmse', xlim=[1,num_epochs],yscale='log')
    print(f'训练log rmse:{float(train_ls[-1]):f}')
    #将网络应用于测试集
    preds = net(test_features).detach().numpy()#net不需要参数，往net里传参数什么意思？
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submission.csv',index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

d2l.plt.show()