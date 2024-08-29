import torch
from d2l import torch as d2l

#多输入单输出通道，输入是n维，那么就有一个n维的核，它们相应维度相乘相加在与不同维度相加就得到了单输出通道
#多输入多输出通道，输入是n维，那么就可有m个n维的核，每个核生成一个通道，就得到了多通道输出
#输入通道数和输出通道数没有太多相关性

#多输入通道单输出通道，只有一个kernel
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

#多输入多输出多输出通道，有多个kernel，但是它们构成矩阵。
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

#1x1卷积层，每一个卷积核的维度和输入的维度是相同的（也可以看作全连接层）
#只对输入通道和输出通道进行融合，不做空间的匹配 
def corr2d_multi_in_out_1x1(X, K):
    #X输入是三维的(通道数，行数，列数)，kerenl是四维的，它的0维就是输出通道数
    c_i, h, w = X.shape
    #c_0输出的维数，也就是输出通道数，它等于kernel的第0维。
    c_o = K.shape[0]
    #X的每一行就是原来的一个面的数据
    X = X.reshape((c_i, h * w))
    #K有四个维度，但是最外面两个维度干有维度但没有数值，就把最外面两个维度拿掉
    #K的每一行就是原来的一个核
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法，Y的每一行就是原来的一个面，得到h*w列
    #之所以用K*X而不是习惯的X*K，是为了便于最后将输出Y恢复到三维，如果用X*K，在最后需要对Y进行转置再恢复为三维
    Y = torch.matmul(K, X)
    ##本来形状是(c_o, h*w)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6