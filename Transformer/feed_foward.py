import torch
import torch.nn as nn


class PositioniseFeedFoward(nn.Module):
    # two linear layers
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(self).__init__()
        # d_model is dimensional of embedding, as well input and output linear layer dim
        # w1 col dim and w2 row dim
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # last layer output
        # w1 -> relu -> w2
        return self.w2(self.dropout(F.relu(self.w1(x)))