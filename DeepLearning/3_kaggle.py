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
#包括数据预处理、模型设计和超参数选择

#建立字典DATA_HUB,将数据集名称的字符串映射到数据集相关的二元组上,二元组包含数据集的url和验证文件完整性的sha-1密钥.
#字典形式{'数据集名'：(url,sha1_hsah)}
#所有类似的数据集都托管在地址为DATA_URL的站点上.
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

train_data = pd.read_csv(dowload('kaggle_house_train'))
test_data = pd.read_csv(dowload('kaggle_house_test'))

#
print(train_data.shape)
print(test_data.shape)