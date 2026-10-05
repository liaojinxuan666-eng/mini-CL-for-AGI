import torch
import torch.nn as nn

from .hybrid import HybridLM


class WorldModel(nn.Module):
    """
    世界模型。复用 HybridLM 主干，只把词表扩到 204 并直接预测 next token。

    token 布局:
        0-99    x 坐标
        100-199 y 坐标
        200-203 动作

    输入:  [x_1, y_1, a_1, x_2, y_2, a_2, ...]
    目标:  token 3, 4, 6, 7, 9, 10, ... 位置的 x/y
    """

    def __init__(
        self,
        vocab_size=204,
        d_model=128,
        n_head=4,
        window=64,
        layer_kinds=None,
        max_len=2048,
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

        from .mamba_block import MambaBlock
        from .attention_block import AttentionBlock

        layers = []
        for k in layer_kinds:
            if k == "attn":
                layers.append(AttentionBlock(d_model, n_head, window, dropout))
            else:
                layers.append(MambaBlock(d_model, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        assert L <= self.max_len
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))