# model/attention_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions):
    B, H, L, Dh = x.shape
    half = Dh // 2
    freqs = torch.exp(
        -torch.arange(0, half, device=x.device).float()
        * (torch.log(torch.tensor(10000.0, device=x.device)) / half)
    )
    angles = positions.float()[:, None] * freqs[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return torch.cat([rot1, rot2], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_head=4, window=64, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window = window

        self.ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        h = self.ln(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        pos = torch.arange(L, device=x.device)
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        idx = torch.arange(L, device=x.device)
        diff = idx[:, None] - idx[None, :]
        mask = (diff < 0) | (diff >= self.window)
        attn = attn.masked_fill(mask[None, None], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return residual + self.dropout(out)