from Transformer.norm import LayerNorm
import norm
import torch
import torch.nn as nn

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size is the dimension of embedding
        super(self).__init_()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size
    
    def forward(self, x, sublayer):
        # x is the output of the last layer tensor
        # sublayer is sublayer function
        # normalize x -> put it into sublayer -> dropout -> residual
        return x + self.dropout(sublayer(self.norm(x)))
