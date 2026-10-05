# model/hybrid.py
import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .attention_block import AttentionBlock


class HybridLM(nn.Module):
    def __init__(
        self,
        vocab_size=34,
        d_model=128,
        n_head=4,
        window=64,
        layer_kinds=None,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        if layer_kinds is None:
            layer_kinds = ["ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"]
        self.layer_kinds = layer_kinds
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layers = []
        for k in layer_kinds:
            if k == "attn":
                layers.append(AttentionBlock(d_model, n_head, window, dropout))
            else:
                layers.append(MambaBlock(d_model, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _trunk(self, x):
        B, L = x.shape
        assert L <= self.max_len
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.ln_f(h)                     # [B, L, d]

    def forward(self, x):
        return self.head(self._trunk(x))

    @torch.no_grad()
    def forward_hidden(self, x):
        """给检索用：只返回最后一层 hidden，不接 head。"""
        return self._trunk(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx