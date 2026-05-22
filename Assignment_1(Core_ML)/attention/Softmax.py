import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
from data import create_dataloader

# Run for different context lengths and throughput


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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float, 
                 pos_encoding_fn=None, use_rope: bool = False, rope_scale: float = 1,
                 rpe_fn = None, use_rpe: bool = False,
                 alibi_fn = None, use_alibi: bool = False, **kwargs):
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
        
        # In case rope is used
        self.use_rope = use_rope
        self.pos_encoding_fn = pos_encoding_fn

        # In case rpe is used
        self.use_rpe = use_rpe
        self.rpe_fn = rpe_fn

        # In case alibi is used
        self.use_alibi = use_alibi
        self.alibi_fn = alibi_fn

        self.rope_scale = rope_scale

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, rpe_bias = None, alibi_pos = None):
        d_k = query.shape[-1]

        # (batch, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = ((query @ key.transpose(-2,-1))/math.sqrt(d_k))
        if rpe_bias is not None:
            attention_scores = attention_scores + rpe_bias
        if alibi_pos is not None:
            attention_scores = attention_scores + alibi_pos

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 0.0)
        attention_scores = torch.relu(attention_scores) # batch, h, seq_len, seq_len

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

        rpe_bias = None        
        alibi_pos = None
        if self.use_rope:
            # print("Successfully reached this fn")   
            query, key = self.pos_encoding_fn(query, key, self.rope_scale)
        elif self.use_rpe:
            seq_len = query.shape[-2]
            rpe_bias = self.rpe_fn(seq_len, query.device)
        elif self.use_alibi:
            seq_len = query.shape[-2]
            alibi_pos = self.alibi_fn(seq_len, query.device)

        # (batch, h, seq_len, d_k) 
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout, 
                                                                rpe_bias=rpe_bias, alibi_pos=alibi_pos)
        
        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.Wo(x)


