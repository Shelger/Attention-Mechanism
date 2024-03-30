import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # feature is dimension of embedding
        super(self).__init__()
        # nn.parameter will let a2 and b2 trained during training
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps
    
    def forward(self, x):
        # x is the last layer output
        mean = x.mean(-1, keepdim=True)
        # for x, last dim standard error, to make sure input and output dim same
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

