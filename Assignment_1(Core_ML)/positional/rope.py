import torch
import math

def rotate_half(x):
    # Rotates half the hidden dims of the input
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k):
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    device = q.device

    position = torch.arange(seq_len, device=device).unsqueeze(1) # Shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, head_dim, 2, device=device) * -(math.log(10000.0) / head_dim))
    freqs = position * div_term  # Shape: (seq_len, head_dim / 2)
    
    emb = torch.cat((freqs, freqs), dim=-1)  # Shape: (seq_len, head_dim)
    
    cos = emb.cos().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)

    # Apply the rotation
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    
    return q_rotated, k_rotated