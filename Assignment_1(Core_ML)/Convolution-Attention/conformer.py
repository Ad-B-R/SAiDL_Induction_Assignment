import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
from data_conv import create_dataloader


# Run for different context lengths

class AlibiPosition(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        m_1 = 2**(-8/self.num_heads)
        slope = torch.pow(m_1, torch.arange(1, self.num_heads+1))

        slope = slope.view(self.num_heads, 1, 1)

        self.register_buffer("slope", slope, persistent=False)

    def forward(self, seq_len, device):
        pos = torch.arange(seq_len, device=device)
        # (seq_len, seq_len)
        distance_matrix = torch.abs(pos[None, :] - pos[:, None])  
        # (1, seq_len, seq_len) -> (h, seq_len, seq_len)
        alibi_pos = -self.slope.view(-1, 1, 1) * distance_matrix   
        # (1, h, seq_len, seq_len)
        return alibi_pos.unsqueeze(0)                               
    
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
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x + self.beta
    
class ResidualConnections(nn.Module):
    def __init__(self, features:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class NGramConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1

        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=0,
            groups=d_model
        )
        self.activation = nn.GELU()

        print(f"Kernel size: {kernel_size}")
        
    def forward(self, x):
        x_transposed = x.transpose(1, 2)                                  
        x_padded = F.pad(x_transposed, (self.left_pad, 0))                
        conv_out = self.conv1d(x_padded)                                  
        out = conv_out.transpose(1, 2)                                    # (batch, seq_len, d_model)
        return self.activation(out)


class StandardAttentionMask(nn.Module):
    def __init__(self, features, layers: nn.ModuleList, window=64, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(layers) if not isinstance(layers, nn.ModuleList) else layers
        self.norm = LayerNorm(features)
        self.window = window

    def forward(self, x, mask=None):
        if mask is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = torch.triu(mask, diagonal=-self.window + 1) 
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float, 
                 alibi_fn = None, **kwargs):  
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model%h == 0,  "d_model not divisible by h"

        self.d_k = d_model//h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.alibi_fn = alibi_fn

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, alibi_pos = None):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = ((query @ key.transpose(-2,-1))/math.sqrt(d_k))
        attention_scores = attention_scores + alibi_pos

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1) # batch, h, seq_len, seq_len

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores


    def forward(self, x, mask):
        # batch, seq len, d_model
        query = self.Wq(x)
        value = self.Wv(x)
        key = self.Wk(x)
        # batch, seq_len, h, d_k -> batch, h, seq_len, d_k
        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1,2)
        # key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).permute([0,2,1,3])
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1,2)

        alibi_pos = None       
        if True:
            seq_len = query.shape[-2]
            alibi_pos = self.alibi_fn(seq_len, query.device)

        # (batch, h, seq_len, d_k) 
        x, self.attention_scores = SlidingWindowAttention.attention(query, key, value, mask, self.dropout, 
                                                                    alibi_pos=alibi_pos)

        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.Wo(x)

class ConvAttentionBlock(nn.Module):
    def __init__(self, features, self_attention_block: nn.Module, feed_forward_block: FeedForwardNetwork,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_foward_network = feed_forward_block
        self.conv_block = NGramConvBlock(d_model=features)
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnections(features, dropout) for _ in range(3)])

    def forward(self, x, src_mask):
        # Residual_connections stores residual blocks, 
        # hence following means ResidualConnections.forward(x, sublayer)
        x = self.residual_connections[0](x, 
            lambda x: self.self_attention_block(x,src_mask))
        # residual_connections__Call__() expects only one input function, hence we do this
        x = self.residual_connections[1](x, self.conv_block)
        x = self.residual_connections[2](x, self.feed_foward_network)
        return x