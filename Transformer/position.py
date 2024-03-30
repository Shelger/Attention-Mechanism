from turtle import forward
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000) -> None:
        super().__init__()
        # dropout 置0率，多少比率让神经网络的神经元失效，max_len句子最大长度
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        
        # 初始化绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变换矩阵div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))

        # 将前面定义的变换矩阵进行一个基数偶数分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将一个二维tensor扩充成三维
        pe = pe.unsqueeze(0)

        # 将为之编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步更新
        # 注册成buffer后，就可以在模型保存后重新加载的时候，将这个位置编码器和模型参数加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe编码太长，将第二个维度，也就是maxlen对应的维度缩小成x的句子长度同等
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# d_model = 512
# dropout = 0.1
# max_len = 60
# div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
# print(torch.arange(0, d_model, 2))

# print(np.triu([[1,3,4],[13,4,76],[11,-1,5]], k=0))
# build mask function:
def subsequent_mask(size):
    # size is the last dimension to form a matrix
    attn_shape = (1,size,size)
    # np.ones() build all 1 tensor, then use np.triu() to build a left-top triangle
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    # reverse triangle
    return torch.from_numpy(1-subsequent_mask)

# size = 5
# sm = subsequent_mask(size)
# print(sm)