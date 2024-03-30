import torch
import torch.nn as nn
from sublayer import SublayerConnection
from norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # input parameter
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        # clones
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, source_mask, target_mask):
        # source_mask is the mask of source data
        # output target mask
        m = memory
        # self multihead attention -> attention -> feedforward
        # target mask to cover future information
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # multihead attention
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))

        # x for feed forward
        x = self.sublayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, decoder_layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, n)
        self.norms = LayerNorm(decoder_layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norms(x)
