from tkinter import Variable
import torch
import torch.nn as nn
import math

from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model嵌入词维度，vocab词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

d_model= 512
vocab = 1000

x = Variable(torch.LongTensor([[7,1,2],[8,2,3]]))
embeddings = Embeddings(d_model, vocab)
print(embeddings(x))

