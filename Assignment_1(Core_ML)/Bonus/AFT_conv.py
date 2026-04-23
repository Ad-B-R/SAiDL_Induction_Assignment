import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
    
class AFTConv(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.0, **kwargs):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        # projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, n_heads)     
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.w = nn.Parameter(torch.zeros(n_heads, window_size))

        self.gamma = nn.Parameter(torch.zeros(n_heads))  
        self.beta = nn.Parameter(torch.zeros(n_heads))   

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.d_head
        S = self.window_size


        q = torch.sigmoid(self.Wq(x))              
        v = self.Wv(x)
        k = self.Wk(x) / math.sqrt(Dh)             

        # reshape heads
        q = q.view(B, T, H, Dh)
        v = v.view(B, T, H, Dh)

        k = torch.clamp(k, -15, 15)
        k_exp = torch.exp(k)                       

        
        kv_global = torch.sum(k_exp.unsqueeze(-1) * v, dim=1, keepdim=True)  
        k_global = torch.sum(k_exp, dim=1, keepdim=True)                     

        w = self.w                                   

        mean = w.mean(dim=1, keepdim=True)
        std = w.std(dim=1, keepdim=True) + 1e-6

        w_norm = (w - mean) / std

        w_hat = (
            self.gamma.unsqueeze(1) * w_norm +
            self.beta.unsqueeze(1)
        )                                           

        kernel = torch.exp(torch.clamp(w_hat, -15, 15)) - 1  

        k_exp_ = k_exp.permute(0, 2, 1).contiguous()       
        kv_ = (k_exp.unsqueeze(-1) * v).permute(0, 2, 3, 1) # (B, H, Dh, T)

        k_exp_ = k_exp_.reshape(B * H, 1, T)
        kv_ = kv_.reshape(B * H * Dh, 1, T)

        # expand kernels
        kernel_k = kernel[0].view(1,1,S)
        kernel_v = kernel[0].view(1,1,S)
        
        k_local = F.conv1d(
            k_exp_,
            kernel_k,
            padding=S-1,
            groups=1
        )[:, :, :T]
        k_local = k_local.view(B, H, T).permute(0, 2, 1)   # (B, T, H)

        kv_local = F.conv1d(
            kv_,
            kernel_v,
            padding=S-1,
            groups=1
        )[:, :, :T]
        kv_local = kv_local.view(B, H, Dh, T).permute(0, 3, 1, 2)  # (B, T, H, Dh)

        num = kv_local + kv_global
        den = k_local.unsqueeze(-1) + k_global.unsqueeze(-1) + 1e-6

        out = q * (num / den)   # (B, T, H, Dh)

        out = out.reshape(B, T, D)
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
