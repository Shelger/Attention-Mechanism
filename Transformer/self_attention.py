
from Transformer.embeddings import Embeddings
import torch
import torch.nn as nn



from cmath import sqrt


def attention(query, key, value, mask=None, dropout=None):
    # tensor dimension [batch size, sequence length, embedding dimension]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # for the last dimension, do softmax
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# clone linear layers
def clones(module, N):
    # module is the cloned one, N is the number to be cloned
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class multiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout = 0.1):
        # head how many heads
        super(self).__init__()
        # assert embedding_dim is divided by head
        assert embedding_dim % head == 0

        # get dim of each head
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim

        # get linears
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value are three attention inputs, mask is mask tensor
        if mask is not None:
            # dimension extension, for no.n head
            mask = mask.unsqueeze(1)
        
        # batch_size
        batch_size = query.size(0)

        # zip connects layers and inputs, output uses view and transpose for dimension
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) for model, x in zip(self.linears, (query, key, value))]

        # input all outputs into attention layer
        x, attn =  attention(query, key, value, mask, self.dropout)

        # need to convert shape into 3 dimension
        # becuase of transpose, we need to turn it back
        # we need to contiguous for view
        x.transpose(1,2).contiguous().view(batch_size, -1, self.d_k * self.head)

        # put x into last linear
        return self.linears[-1](x)
    
# head = 8
# embedding_dim = 512
# dropout = 0.2
# query = key = value = pe_result
