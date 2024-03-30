


from statistics import mode
from Transformer.self_attention import multiHeadAttention


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        # generator output
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        # source data, target data
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target, target_mask)
    

def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    # d_ff feedforward matrix dimension
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, droupout)
    position = PositionEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
                            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                            nn.Sequential(Embedding(d_model, source_vocab), c(position)),
                            nn.Sequential(Embedding(d_model, target_vocab), c(position)),
                            Generator(d_model, target_vocab))
    # initialized model coefficients, check if dimension > 1, average sparate matrix
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model