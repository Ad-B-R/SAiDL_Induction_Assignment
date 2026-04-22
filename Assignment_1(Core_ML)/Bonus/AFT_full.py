import torch
import torch.nn as nn
import math
from data import create_dataloader

# Run for different context lengths and throughput

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(self.seq_len, d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* 
                            (-math.log(10000)/self.d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    @torch.no_grad()
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :] # for different broadcasting purposes
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features:int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean)/(std + self.eps) + self.beta
    
class StandardAttention(nn.Module):
    def __init__(self, features, layers: nn.ModuleList, **kwargs):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)
    
    def forward(self, x, mask=None):
        if mask is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class AFTFull(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 2048, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = seq_len
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        # Learned pairwise position biases (T_max, T_max)
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, seq_len))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = torch.sigmoid(self.Wq(x))               
        k_logits = self.Wk(x) / math.sqrt(D)           
        k_logits = torch.clamp(k_logits, -20, 20)      
        k_exp = torch.exp(k_logits)                    

        v = self.Wv(x)                                 
        kv = k_exp * v                                 

        # 2. Positional bias
        w = self.pos_bias[:T, :T]                      

        if mask is not None:
            mask = mask.squeeze(1)                     
            w = w.unsqueeze(0).expand(B, T, T)
            w = w.masked_fill(mask == 0, float('-inf'))
        else:
            w = w.unsqueeze(0).expand(B, T, T)

        w = torch.clamp(w, -10, 10)
        w_exp = torch.exp(w)                           

        num = torch.einsum('bti,bid->btd', w_exp, kv)
        den = torch.einsum('bti,bid->btd', w_exp, k_exp) + 1e-6

        # 6. Output
        out = q * (num / den)
        out = self.dropout(out)

        return self.Wo(out)
    
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fnn1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fnn2(self.dropout(torch.relu(self.fnn1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features:int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean)/(std + self.eps) + self.beta

class ResidualConnections(nn.Module):
    def __init__(self, features:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class AttentionBlock(nn.Module):
    def __init__(self, features, self_attention_block: nn.Module, feed_forward_block: FeedForwardNetwork,
                 dropout: float, ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_foward_network = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnections(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Residual_connections stores residual blocks, 
        # hence following means ResidualConnections.forward(x, sublayer)
        x = self.residual_connections[0](x, 
            lambda x: self.self_attention_block(x,src_mask))
        # residual_connections__Call__() expects only one input function, hence we do this
        x = self.residual_connections[1](x, self.feed_foward_network)
        return x
