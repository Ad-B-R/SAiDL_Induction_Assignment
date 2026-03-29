import torch
import torch.nn as nn


import torch
import torch.nn as nn

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
