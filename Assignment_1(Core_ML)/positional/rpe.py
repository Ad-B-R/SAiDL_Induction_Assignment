import torch
import torch.nn as nn

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # distances range: [-(L-1), ..., +(L-1)]
        self.num_buckets = 2 * max_seq_len - 1

        self.bias = nn.Parameter(torch.zeros(num_heads, self.num_buckets))

    def forward(self, seq_len, device):
        pos = torch.arange(seq_len, device=device)

        # (seq_len, seq_len)
        rel_pos = pos[None, :] - pos[:, None]

        rel_pos += self.max_seq_len - 1
        rel_pos = rel_pos.clamp(0, self.num_buckets - 1)
        bias = self.bias[:, rel_pos]
        # (1, h, seq_len, seq_len)
        return bias.unsqueeze(0)
    
    