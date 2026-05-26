import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import math
import conformer

class ConvOnlyBlock(nn.Module):
    def __init__(self, features, feed_forward_block, dropout):
        super().__init__()
        # 1D Convolution for extracting local n-grams
        self.conv_block = conformer.NGramConvBlock(d_model=features)
        self.feed_forward_network = feed_forward_block
        self.residual_connections = nn.ModuleList([conformer.ResidualConnections(features, dropout) for _ in range(2)])

    def forward(self, x, mask=None): 
        x = self.residual_connections[0](x, self.conv_block)
        x = self.residual_connections[1](x, self.feed_forward_network)
        return x

class ConvAttentionBlock(nn.Module):
    def __init__(self, features, self_attention_block: nn.Module, feed_forward_block, dropout):
        super().__init__()
        self.conv_block = conformer.NGramConvBlock(d_model=features)
        self.feed_forward_network = feed_forward_block
        self.residual_connections = nn.ModuleList([conformer.ResidualConnections(features, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.residual_connections[0](x, self.conv_block)
        x = self.residual_connections[1](x, self.feed_forward_network)
        return x